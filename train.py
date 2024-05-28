import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from tqdm import tqdm 
from utils import parse_args, collate_fn, MyDataset 
from attention import Consistent_IPAttProcessor, Consistent_AttProcessor
from attention import FacialEncoder
from functions import ProjPlusModel, BalancedL1Loss, unet_store_cross_attention_scores, get_object_localization_loss 

exp_name = 'ConsistentID' 
initial_epoch = 0

class ConsistentID(torch.nn.Module):
    """ConsistentID"""
    def __init__(self, unet, image_proj_model, adapter_modules, facial_encoder):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.FacialEncoder = facial_encoder

        ### attention loss
        self.cross_attention_scores = {}
        self.localization_layers  = 5
        self.facial_weight = 0.01
        self.mask_loss_prob = 0.5
        self.unet = unet_store_cross_attention_scores(
            self.unet, self.cross_attention_scores, self.localization_layers 
        )
        self.object_localization_loss_fn = BalancedL1Loss(threshold=1.0, normalize=True)    

    def forward(self, noisy_latents, timesteps, prompt_embeds, image_embeds, faceid_embeds, \
                multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask, \
                noise, parsing_mask_lists, facial_masks, facial_token_idxs, facial_token_idx_masks): 

        ### Overall Feature
        faceid_tokens = self.image_proj_model(faceid_embeds, image_embeds) 

        ### Fine-grained Feature
        prompt_id_embeds = self.FacialEncoder(prompt_embeds, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask) 

        ### Final Features
        prompt_id_embeds = torch.cat([prompt_id_embeds, faceid_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, prompt_id_embeds).sample 
        
        #### Random Mask     
        target=noise
        pred=noise_pred
        loss_dict = {"background_loss": 0}
        if torch.rand(1) < self.mask_loss_prob:
            try:
                mask_list = [TF.to_tensor(image['WithoutBackground']).unsqueeze(0) for image in parsing_mask_lists] 
                mask_stacked = torch.cat(mask_list, dim=0)
                mask_final = F.interpolate(mask_stacked,size=(pred.shape[-2], pred.shape[-1]),mode="bilinear",align_corners=False,) 
                pred = pred * mask_final.to(pred.device, dtype=pred.dtype)
                target = target * mask_final.to(target.device, dtype=target.dtype)

                background_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
                loss_dict["background_loss"] = background_loss
            except:
                print(f"The fail 'Background' of parsing_mask_lists: {parsing_mask_lists}")

        predict_loss = F.mse_loss(pred.float(), target.float(), reduction="mean") 
        loss_dict["predict_loss"] = predict_loss
        
        ### Attention Loss
        loss_dict["facial_loss"] = 0
        object_segmaps = facial_masks
        image_token_idx = facial_token_idxs
        image_token_idx_mask = facial_token_idx_masks
        facial_loss = get_object_localization_loss(
            self.cross_attention_scores,
            object_segmaps,
            image_token_idx,
            image_token_idx_mask,
            self.object_localization_loss_fn,
        )
            
        facial_loss = self.facial_weight * facial_loss 
        loss_dict["facial_loss"]=facial_loss

        return pred, loss_dict

def main():

    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Projection
    image_proj_model = ProjPlusModel(
        cross_attention_dim=768,
        id_embeddings_dim=512,
        clip_embeddings_dim=image_encoder.config.hidden_size,
        num_tokens=args.num_tokens,
    )

    ### Facial Encoder
    facial_encoder = FacialEncoder(image_CLIPModel_encoder=None)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "fp32": ### TODO
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype) 
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype) 
    image_proj_model.to(accelerator.device, dtype=weight_dtype) 

    # init adapter modules
    lora_rank = 128
    attn_procs = {}
    unet_sd = unet.state_dict()

    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim 
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]  
        if cross_attention_dim is None:  
            attn_procs[name] = Consistent_AttProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)  
        else: 
            layer_name = name.split(".processor")[0] 
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"], 
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"], 
            }
            attn_procs[name] = Consistent_IPAttProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)  
            attn_procs[name].load_state_dict(weights, strict=False)

    unet.set_attn_processor(attn_procs) 
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values()) 
    
    adapter_modules.requires_grad_(False)
    consistentID_model = ConsistentID(unet, image_proj_model, adapter_modules, facial_encoder)

    optimizer_cls = torch.optim.AdamW
    unet_params = list([p for p in consistentID_model.unet.parameters() if p.requires_grad]) 
    other_params = list( 
        [p for n, p in consistentID_model.named_parameters() if p.requires_grad and "unet" not in n]
    )

    optimizer = optimizer_cls( 
        [
            {"params": unet_params, "lr": 1e-4*1.0 },
            {"params": other_params, "lr": 1e-4},
        ],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # dataloader
    train_dataset = MyDataset(args.data_json_file, args.data_json_mutiID_file, tokenizer=tokenizer, size=args.resolution, \
                              image_root_path=args.data_root_path, faceid_root_path=args.faceid_root_path, parsing_root_path=args.parsing_root_path)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    consistentID_model, optimizer, train_dataloader = accelerator.prepare(consistentID_model, optimizer, train_dataloader)

    for epoch in range(initial_epoch, args.num_train_epochs):
        begin = time.perf_counter()
        global_step = 0
        progress_bar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(train_dataloader), disable=not accelerator.is_local_main_process,)

        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            
            lora_rank = 128
            attn_procs = {}
            unet_sd = unet.state_dict()
            
            with accelerator.accumulate(consistentID_model):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    clip_images = batch["clip_images"]
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]

                    hidden_states = []
                    facial_clip_images = batch["facial_clip_images"]                    
                    for facial_clip_image in facial_clip_images:
                        hidden_state = image_encoder(facial_clip_image.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                        hidden_states.append(hidden_state)
                    multi_facial_embeds = torch.stack(hidden_states)

                with torch.no_grad(): 
                    clean_input_ids = batch["clean_input_ids"]
                    prompt_embeds = text_encoder(clean_input_ids.to(accelerator.device))[0]

                faceid_embeds = batch["face_id_embeds"].to(accelerator.device, dtype=weight_dtype)

                facial_token_masks = batch["facial_token_masks"]
                valid_facial_token_idx_mask = batch["facial_token_idx_masks"] 

                parsing_mask_lists = batch["parsing_mask_lists"]
                facial_masks = batch["facial_masks"] 
                facial_token_idxs = batch["facial_token_idxs"]
                facial_token_idx_masks = batch["facial_token_idx_masks"]

                noise_pred, loss_dict = consistentID_model(noisy_latents, timesteps, prompt_embeds, image_embeds, faceid_embeds, \
                            multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask,
                            noise, parsing_mask_lists, facial_masks, facial_token_idxs, facial_token_idx_masks)

                predict_loss = loss_dict["predict_loss"] 
                facial_loss = loss_dict["facial_loss"]

                background_loss = loss_dict["background_loss"]

                loss = predict_loss + facial_loss
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()  
                      
                # Backpropagate
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}, predict_loss: {}, facial_loss: {}, background_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss, predict_loss, facial_loss, background_loss))
                    
            global_step += 1 
            
            progress_bar.set_description(f"{exp_name}Epoch {epoch + 1}/{args.num_train_epochs} - Step {step}/{len(train_dataloader)}")
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"{exp_name}_Epoch{epoch+1}-{global_step}")
                accelerator.save_state(save_path)
                
            begin = time.perf_counter()
                     
if __name__ == "__main__":
    main()    

