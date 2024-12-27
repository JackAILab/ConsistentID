import os
from pathlib import Path
import itertools
import time
import torch
import torch.nn.functional as F
import math
import torchvision.transforms as T
import gc

from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

import torchvision.transforms.functional as TF

from tqdm import tqdm 
import torch.nn as nn 

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils_SDXL import parse_args, collate_fn, MyDataset
from attention import FacialEncoder, Consistent_IPAttProcessor, Consistent_AttProcessor
from functions import MLPProjModel, ProjPlusModel, BalancedL1Loss, unet_store_cross_attention_scores, get_object_localization_loss

# exp_name
exp_name = 'ConsistentID_SDXL'
initial_epoch = 0

class ConsistentID(torch.nn.Module):
    """ConsistentID"""
    def __init__(self, unet, image_proj_model, adapter_modules, image_CLIPModel_encoder=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        self.FacialEncoder = FacialEncoder(image_CLIPModel_encoder)
        self.cross_attention_scores = {}
        self.localization_layers  = 3
        self.facial_weight = 0.01
        self.mask_loss_prob = 0.5

        self.unet = unet_store_cross_attention_scores( 
            self.unet, self.cross_attention_scores, self.localization_layers 
        )

        self.object_localization_loss_fn = BalancedL1Loss(threshold=1.0, normalize=True)    
        # self.load_from_checkpoint(ckpt_path="./ConsistentID.bin")

    def forward(self, noisy_latents, timesteps, prompt_embeds, image_embeds, faceid_embeds, \
                unet_added_cond_kwargs, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask, \
                noise, batch, parsing_mask_lists, facial_masks, facial_token_idxs, facial_token_idx_masks): 

        faceid_tokens = self.image_proj_model(faceid_embeds, image_embeds)

        prompt_id_embeds = self.FacialEncoder(prompt_embeds, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)

        prompt_id_embeds = torch.cat([prompt_id_embeds, faceid_tokens], dim=1)

        noise_pred = self.unet(noisy_latents, timesteps, prompt_id_embeds, added_cond_kwargs=unet_added_cond_kwargs).sample

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
    
    def load_from_checkpoint(self, ckpt_path: str):

        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        orig_FacialEncoder_sum = torch.sum(torch.stack([torch.sum(p) for p in self.FacialEncoder.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["adapter_modules"], strict=True)
        self.FacialEncoder.load_state_dict(state_dict["FacialEncoder"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        new_FacialEncoder_sum = torch.sum(torch.stack([torch.sum(p) for p in self.FacialEncoder.parameters()]))

        # # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
        assert orig_FacialEncoder_sum != new_FacialEncoder_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

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

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    image_proj_model = ProjPlusModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        clip_embeddings_dim=image_encoder.config.hidden_size,
        num_tokens=args.num_tokens,
    )

    weight_dtype = torch.float16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "fp32":
        weight_dtype = torch.float32
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp8":
        weight_dtype = torch.float8        
    elif accelerator.mixed_precision == 'no':
        weight_dtype = torch.float32  

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_proj_model.to(accelerator.device, dtype=weight_dtype)

    # init adapter modules
    lora_rank = 128 ### important TODO
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

    consistentID_model = ConsistentID(unet, image_proj_model, adapter_modules)

    optimizer_cls = torch.optim.AdamW

    unet_params = list([p for p in consistentID_model.unet.parameters() if p.requires_grad])
    other_params = list(
        [p for n, p in consistentID_model.named_parameters() if p.requires_grad and "unet" not in n]
    )
    parameters = unet_params + other_params

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
    train_dataset = MyDataset(args.data_json_file, args.data_json_mutiID_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, \
                              image_root_path=args.data_root_path, faceid_root_path=args.faceid_root_path, parsing_root_path=args.parsing_root_path)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    consistentID_model, optimizer, train_dataloader = accelerator.prepare(consistentID_model, optimizer, train_dataloader)

    # # Train
    for epoch in range(initial_epoch, args.num_train_epochs):
        begin = time.perf_counter()
        global_step = 0
        progress_bar = tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch + 1}/{args.num_train_epochs}", total=len(train_dataloader), disable=not accelerator.is_local_main_process,)

        for step, batch in enumerate(train_dataloader):
            if any("error" in item for item in batch):
                print("Skipping batch with invalid data")
                continue            
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(consistentID_model):
                with torch.no_grad():
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    clip_images = batch["clip_images"]
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2] # .image_embeds

                    # level 3
                    hidden_states = []
                    facial_clip_images = batch["facial_clip_images"]                    
                    for facial_clip_image in facial_clip_images:
                        hidden_state = image_encoder(facial_clip_image.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                        hidden_states.append(hidden_state)
                    multi_facial_embeds = torch.stack(hidden_states)

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_id_alls'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_id_all2s'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    prompt_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat     
                
                add_time_ids = [
                    batch["original_sizes"].to(accelerator.device),
                    batch["crop_coords_top_lefts"].to(accelerator.device),
                    batch["target_sizes"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}

                faceid_embeds = batch["face_id_embeds"].to(accelerator.device, dtype=weight_dtype)

                # level 3
                facial_token_masks = batch["facial_token_masks"]
                valid_facial_token_idx_mask = batch["facial_token_idx_masks"]

                parsing_mask_lists = batch["parsing_mask_lists"]
                facial_masks = batch["facial_masks"]
                facial_token_idxs = batch["facial_token_idxs"]
                facial_token_idx_masks = batch["facial_token_idx_masks"]

                noise_pred, loss_dict = consistentID_model(noisy_latents, timesteps, prompt_embeds, image_embeds, faceid_embeds, \
                            unet_added_cond_kwargs, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask,
                            noise, batch, parsing_mask_lists, facial_masks, facial_token_idxs, facial_token_idx_masks)

                predict_loss = loss_dict["predict_loss"]
                facial_loss = loss_dict["facial_loss"]

                background_loss = loss_dict["background_loss"]

                loss = predict_loss + facial_loss

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
                if not os.path.exists(save_path):
                    try:
                        os.makedirs(save_path)
                    except:
                        print(f"The path can not be make {save_path}!!!")
                
                save_path_pth = save_path + "/ConsistentID_SDXL.pth"
                torch.save(consistentID_model.state_dict(), save_path_pth)

            if global_step == len(train_dataloader)-1: # save the lastest
                save_path_pth = os.path.join(args.output_dir, f"{exp_name}_checkpoint-lasted") + "/ConsistentID_SDXL.pth"
                if not os.path.exists(os.path.dirname(save_path_pth)):
                    try:
                        os.makedirs(os.path.dirname(save_path_pth))
                    except:
                        print(f"The path can not be make {save_path}!!!")                
                
                torch.save(consistentID_model.state_dict(), save_path_pth) 

            begin = time.perf_counter()

                     
if __name__ == "__main__":
    main()    

















