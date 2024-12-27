from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import cv2
import PIL
import numpy as np 
from PIL import Image
import torch
from torchvision import transforms
from insightface.app import FaceAnalysis 
### insight-face installation can be found at https://github.com/deepinsight/insightface
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.utils import _get_model_file
from functions import process_text_with_markers, masks_for_unique_values, fetch_mask_raw_image, tokenize_and_mask_noun_phrases_ends, prepare_image_token_idx
from functions import ProjPlusModel, masks_for_unique_values
from attention import Consistent_IPAttProcessor, Consistent_AttProcessor, FacialEncoder
### Model can be imported from https://github.com/zllrunning/face-parsing.PyTorch?tab=readme-ov-file
### We use the ckpt of 79999_iter.pth: https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812
### Thanks for the open source of face-parsing model.
from models.BiSeNet.model import BiSeNet # resnet tensorflow
import pdb
######################################
########## add for sdxl
######################################
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
######################################
########## add for llava
######################################
# import sys
# sys.path.append("./Llava1.5/LLaVA")
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path
# from llava.eval.run_llava import eval_model

PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]


class ConsistentIDStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    
    @validate_hf_hub_args
    def load_ConsistentID_model(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word_ID: str = '<|image|>',
        trigger_word_facial: str = '<|facial|>',
        image_encoder_path: str = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',   # Import CLIP pretrained model
        bise_net_cp: str = 'JackAILab/ConsistentID/face_parsing.pth',
        torch_dtype = torch.float16,
        num_tokens = 4,
        lora_rank= 128,
        **kwargs,
    ):
        self.lora_rank = lora_rank 
        self.torch_dtype = torch_dtype
        self.num_tokens = num_tokens
        self.set_ip_adapter()
        self.image_encoder_path = image_encoder_path
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=self.torch_dtype
        )   
        self.clip_image_processor = CLIPImageProcessor()
        self.id_image_processor = CLIPImageProcessor()
        self.crop_size = 512

        # FaceID
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) ### root="/root/.insightface/models/buffalo_l"
        self.app.prepare(ctx_id=0, det_size=(512, 512)) ### (640, 640)

        ### BiSeNet
        self.bise_net = BiSeNet(n_classes = 19)
        self.bise_net.cuda()
        self.bise_net_cp= bise_net_cp # Import BiSeNet model
        self.bise_net.load_state_dict(torch.load(self.bise_net_cp)) # , map_location="cpu"
        self.bise_net.eval()
        # Colors for all 20 parts
        self.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 0, 85], [255, 0, 170],
                    [0, 255, 0], [85, 255, 0], [170, 255, 0],
                    [0, 255, 85], [0, 255, 170],
                    [0, 0, 255], [85, 0, 255], [170, 0, 255],
                    [0, 85, 255], [0, 170, 255],
                    [255, 255, 0], [255, 255, 85], [255, 255, 170],
                    [255, 0, 255], [255, 85, 255], [255, 170, 255],
                    [0, 255, 255], [85, 255, 255], [170, 255, 255]]
        
        ### LLVA Optional
        self.llva_model_path = "" #TODO import llava weights
        self.llva_prompt = "Describe this person's facial features for me, including face, ears, eyes, nose, and mouth." 
        self.llva_tokenizer, self.llva_model, self.llva_image_processor, self.llva_context_len = None,None,None,None #load_pretrained_model(self.llva_model_path)

        self.FacialEncoder = FacialEncoder(self.image_encoder, embedding_dim=1280, output_dim=2048, embed_dim=2048).to(self.device, dtype=self.torch_dtype)

        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj_model": {}, "adapter_modules": {}, "FacialEncoder": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("unet"):
                            pass
                        elif key.startswith("image_proj_model"):
                            state_dict["image_proj_model"][key.replace("image_proj_model.", "")] = f.get_tensor(key)
                        elif key.startswith("adapter_modules"):   
                            state_dict["adapter_modules"][key.replace("adapter_modules.", "")] = f.get_tensor(key)
                        elif key.startswith("FacialEncoder"):
                            state_dict["FacialEncoder"][key.replace("FacialEncoder.", "")] = f.get_tensor(key)    
            else:
                state_dict = torch.load(model_file, map_location="cuda")
        else:
            state_dict = pretrained_model_name_or_path_or_dict
        

        self.trigger_word_ID = trigger_word_ID
        self.trigger_word_facial = trigger_word_facial

        self.image_proj_model = ProjPlusModel(
            cross_attention_dim=self.unet.config.cross_attention_dim, 
            id_embeddings_dim=512,
            clip_embeddings_dim=self.image_encoder.config.hidden_size, 
            num_tokens=self.num_tokens,  # 4
        ).to(self.device, dtype=self.torch_dtype)
        self.image_proj_model.load_state_dict(state_dict["image_proj_model"], strict=True)

        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["adapter_modules"], strict=True)
        self.FacialEncoder.load_state_dict(state_dict["FacialEncoder"], strict=True)
        print(f"Successfully loaded weights from checkpoint")

        # Add trigger word token
        if self.tokenizer is not None: 
            self.tokenizer.add_tokens([self.trigger_word_ID], special_tokens=True)
            self.tokenizer.add_tokens([self.trigger_word_facial], special_tokens=True)

        ######################################
        ########## add for sdxl
        ######################################
        ### (1) load lora into models
        # print(f"Loading ConsistentID components lora_weights from [{pretrained_model_name_or_path_or_dict}]")
        # self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        ### (2) Add trigger word token for tokenizer_2
        self.tokenizer_2.add_tokens([self.trigger_word_ID], special_tokens=True)

    def set_ip_adapter(self):
        unet = self.unet
        attn_procs = {}
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
                attn_procs[name] = Consistent_AttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=self.lora_rank,
                ).to(self.device, dtype=self.torch_dtype)
            else:
                attn_procs[name] = Consistent_IPAttProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=self.lora_rank, num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.torch_dtype)
        
        unet.set_attn_processor(attn_procs)

    @torch.inference_mode()
    def get_facial_embeds(self, prompt_embeds, negative_prompt_embeds, facial_clip_images, facial_token_masks, valid_facial_token_idx_mask):
        
        hidden_states = []
        uncond_hidden_states = []
        for facial_clip_image in facial_clip_images:
            hidden_state = self.image_encoder(facial_clip_image.to(self.device, dtype=self.torch_dtype), output_hidden_states=True).hidden_states[-2]
            uncond_hidden_state = self.image_encoder(torch.zeros_like(facial_clip_image, dtype=self.torch_dtype).to(self.device), output_hidden_states=True).hidden_states[-2]
            hidden_states.append(hidden_state)
            uncond_hidden_states.append(uncond_hidden_state)
        multi_facial_embeds = torch.stack(hidden_states)       
        uncond_multi_facial_embeds = torch.stack(uncond_hidden_states)   

        # condition 
        facial_prompt_embeds = self.FacialEncoder(prompt_embeds, multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)  

        # uncondition 
        uncond_facial_prompt_embeds = self.FacialEncoder(negative_prompt_embeds, uncond_multi_facial_embeds, facial_token_masks, valid_facial_token_idx_mask)  
        
        return facial_prompt_embeds, uncond_facial_prompt_embeds    

    @torch.inference_mode()   
    def get_image_embeds(self, faceid_embeds, face_image, s_scale=1.0, shortcut=False):

        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.torch_dtype)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)
        image_prompt_tokens = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=shortcut, scale=s_scale)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=shortcut, scale=s_scale)

        return image_prompt_tokens, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, Consistent_IPAttProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_prepare_faceid(self, input_image_path=None):
        faceid_image = cv2.imread(input_image_path)
        face_info = self.app.get(faceid_image)
        if face_info==[]:
            faceid_embeds = torch.zeros_like(torch.empty((1, 512)))
        else:
            faceid_embeds = torch.from_numpy(face_info[0].normed_embedding).unsqueeze(0)

        # print(f"faceid_embeds is : {faceid_embeds}")
        return faceid_embeds

    @torch.inference_mode()
    def parsing_face_mask(self, raw_image_refer):

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        to_pil = transforms.ToPILImage()

        with torch.no_grad():
            ### change sdxl
            image = raw_image_refer.resize((1280, 1280), Image.BILINEAR)
            image_resize_PIL = image
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.float().cuda()
            out = self.bise_net(img)[0]
            parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)
        
        im = np.array(image_resize_PIL)
        vis_im = im.copy().astype(np.uint8)
        stride=1
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1): # num_of_class=17 pi=1~16
            index = np.where(vis_parsing_anno == pi) 
            vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi] 

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        vis_parsing_anno_color = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

        return vis_parsing_anno_color, vis_parsing_anno

    @torch.inference_mode()
    def get_prepare_llva_caption(self, input_image_file, model_path=None, prompt=None):
        
        ### Optional: Use the LLaVA
        # args = type('Args', (), {
        #     "model_path": self.llva_model_path,
        #     "model_base": None,
        #     "model_name": get_model_name_from_path(self.llva_model_path),
        #     "query": self.llva_prompt,
        #     "conv_mode": None,
        #     "image_file": input_image_file,
        #     "sep": ",",
        #     "temperature": 0,
        #     "top_p": None,
        #     "num_beams": 1,
        #     "max_new_tokens": 512
        # })() 
        # face_caption = eval_model(args, self.llva_tokenizer, self.llva_model, self.llva_image_processor)

        ### Use built-in template
        face_caption = "The person has one face, one nose, two eyes, two ears, and a mouth."

        return face_caption

    @torch.inference_mode()
    def get_prepare_facemask(self, input_image_file):

        vis_parsing_anno_color, vis_parsing_anno = self.parsing_face_mask(input_image_file)
        parsing_mask_list = masks_for_unique_values(vis_parsing_anno) 

        key_parsing_mask_list = {}
        key_list = ["Face", "Left_Ear", "Right_Ear", "Left_Eye", "Right_Eye", "Nose", "Upper_Lip", "Lower_Lip"]
        processed_keys = set()
        for key, mask_image in parsing_mask_list.items():
            if key in key_list:
                if "_" in key:
                    prefix = key.split("_")[1]
                    if prefix in processed_keys:                   
                        continue
                    else:            
                        key_parsing_mask_list[key] = mask_image 
                        processed_keys.add(prefix)  
            
                key_parsing_mask_list[key] = mask_image            

        return key_parsing_mask_list, vis_parsing_anno_color

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        face_caption: str,
        key_parsing_mask_list = None,
        image_token = "<|image|>", 
        facial_token = "<|facial|>",
        max_num_facials = 5,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        # pdb.set_trace()
        face_caption_align, key_parsing_mask_list_align = process_text_with_markers(face_caption, key_parsing_mask_list) 
        
        prompt_face = prompt + "; Detail:" + face_caption_align

        max_text_length=330      
        if len(self.tokenizer(prompt_face, max_length=self.tokenizer.model_max_length, padding="max_length",truncation=False,return_tensors="pt").input_ids[0])!=77:
            prompt_face = "; Detail:" + face_caption_align + " Caption:" + prompt
        
        if len(face_caption)>max_text_length:
            prompt_face = prompt
            face_caption_align =  ""

        prompt_text_only = prompt_face.replace("<|facial|>", "").replace("<|image|>", "")
        tokenizer = self.tokenizer
        facial_token_id = tokenizer.convert_tokens_to_ids(facial_token)
        image_token_id = None

        clean_input_id, image_token_mask, facial_token_mask = tokenize_and_mask_noun_phrases_ends(
        prompt_face, image_token_id, facial_token_id, tokenizer) 

        image_token_idx, image_token_idx_mask, facial_token_idx, facial_token_idx_mask = prepare_image_token_idx(
            image_token_mask, facial_token_mask, num_id_images, max_num_facials )

        ######################################
        ########## add for sdxl
        ######################################
        tokenizer_2 = self.tokenizer_2
        facial_token_id2 = tokenizer.convert_tokens_to_ids(facial_token)
        image_token_id2 = None
        clean_input_id2, image_token_mask2, facial_token_mask2 = tokenize_and_mask_noun_phrases_ends(
        prompt_face, image_token_id2, facial_token_id2, tokenizer_2) 

        image_token_idx2, image_token_idx_mask2, facial_token_idx2, facial_token_idx_mask2 = prepare_image_token_idx(
            image_token_mask2, facial_token_mask2, num_id_images, max_num_facials )

        return prompt_text_only, clean_input_id, clean_input_id2, key_parsing_mask_list_align, facial_token_mask, facial_token_idx, facial_token_idx_mask

    @torch.inference_mode()
    def get_prepare_clip_image(self, input_image_file, key_parsing_mask_list, image_size=512, max_num_facials=5, change_facial=True):
        
        facial_mask = []
        facial_clip_image = []
        transform_mask = transforms.Compose([transforms.CenterCrop(size=image_size), transforms.ToTensor(),])
        clip_image_processor = CLIPImageProcessor()

        num_facial_part = len(key_parsing_mask_list)

        for key in key_parsing_mask_list:
            key_mask=key_parsing_mask_list[key]
            facial_mask.append(transform_mask(key_mask))
            key_mask_raw_image = fetch_mask_raw_image(input_image_file,key_mask)
            parsing_clip_image = clip_image_processor(images=key_mask_raw_image, return_tensors="pt").pixel_values
            facial_clip_image.append(parsing_clip_image)

        padding_ficial_clip_image = torch.zeros_like(torch.zeros([1, 3, 224, 224]))
        padding_ficial_mask = torch.zeros_like(torch.zeros([1, image_size, image_size]))

        if num_facial_part < max_num_facials:
            facial_clip_image += [torch.zeros_like(padding_ficial_clip_image) for _ in range(max_num_facials - num_facial_part) ]
            facial_mask += [ torch.zeros_like(padding_ficial_mask) for _ in range(max_num_facials - num_facial_part)]

        facial_clip_image = torch.stack(facial_clip_image, dim=1).squeeze(0)
        facial_mask = torch.stack(facial_mask, dim=0).squeeze(dim=1)

        return facial_clip_image, facial_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        face_caption: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        original_size: Optional[Tuple[int, int]] = None,
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        input_id_images: PipelineImageInput = None,
        input_image_path: PipelineImageInput = None,
        start_merge_step: int = 0,
        class_tokens_mask: Optional[torch.LongTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        ### add for sdxl
        negative_prompt_2: Optional[Union[str, List[str]]] = None,   
        prompt_2: Optional[Union[str, List[str]]] = None,     
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,        
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,   
        guidance_rescale: float = 7.5             
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     height,
        #     width,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        # )

        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale >= 1.0
        input_image_file = input_id_images[0]

        faceid_embeds = self.get_prepare_faceid(input_image_path=input_image_path)
        face_caption = self.get_prepare_llva_caption(input_image_file=input_image_file)
        key_parsing_mask_list, vis_parsing_anno_color = self.get_prepare_facemask(input_image_file)

        assert do_classifier_free_guidance

        # 3. Encode input prompt
        num_id_images = len(input_id_images)

        (
            prompt_text_only,
            clean_input_id,
            clean_input_id2, ### add for sdxl
            key_parsing_mask_list_align,
            facial_token_mask,
            facial_token_idx,
            facial_token_idx_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt = prompt,
            face_caption = face_caption,
            key_parsing_mask_list=key_parsing_mask_list,
            device=device,
            max_num_facials = 5,
            num_id_images= num_id_images,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        text_embeds = self.text_encoder(clean_input_id.to(device), output_hidden_states=True).hidden_states[-2]
        ######################################
        ########## add for sdxl : add pooled_text_embeds
        ######################################
        ### (4-1)
        encoder_output_2 = self.text_encoder_2(clean_input_id2.to(device), output_hidden_states=True)
        pooled_text_embeds = encoder_output_2[0]
        text_embeds_2 = encoder_output_2.hidden_states[-2]
        
        ### (4-2)
        encoder_hidden_states = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat  

        ### (4-3)
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_text_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=self.torch_dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0) ### add_time_ids.Size([2, 6])
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        
        ######################################
        ########## add for sdxl : add pooled_prompt_embeds
        ######################################
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )        
        (
            prompt_embeds,
            negative_prompt_embeds, 
            pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds,  
        )= self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, 
            lora_scale=text_encoder_lora_scale,       
        )

        # 5. Prepare the input ID images
        prompt_tokens_faceid, uncond_prompt_tokens_faceid = self.get_image_embeds(faceid_embeds, face_image=input_image_file, s_scale=1.0, shortcut=True)

        facial_clip_image, facial_mask = self.get_prepare_clip_image(input_image_file, key_parsing_mask_list_align, image_size=1280, max_num_facials=5)
        facial_clip_images = facial_clip_image.unsqueeze(0).to(device, dtype=self.torch_dtype)
        facial_token_mask = facial_token_mask.to(device)
        facial_token_idx_mask = facial_token_idx_mask.to(device)

        cross_attention_kwargs = {}

        # 6. Get the update text embedding
        prompt_embeds_facial, uncond_prompt_embeds_facial = self.get_facial_embeds(encoder_hidden_states, negative_prompt_embeds, \
                                                            facial_clip_images, facial_token_mask, facial_token_idx_mask)

        ########## text_facial embeds
        prompt_embeds_facial = torch.cat([prompt_embeds_facial, prompt_tokens_faceid], dim=1)
        negative_prompt_embeds_facial = torch.cat([uncond_prompt_embeds_facial, uncond_prompt_tokens_faceid], dim=1)

        ########## text_only embeds
        prompt_embeds_text_only = torch.cat([prompt_embeds, prompt_tokens_faceid], dim=1)
        negative_prompt_embeds_text_only = torch.cat([negative_prompt_embeds, uncond_prompt_tokens_faceid], dim=1)

        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                ######################################
                ########## add for sdxl : add unet_added_cond_kwargs
                ######################################                  
                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds_text_only, prompt_embeds_text_only], dim=0
                    )
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0)
                else:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds_facial, prompt_embeds_facial], dim=0
                    )
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_text_embeds], dim=0)

                unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    # return_dict=False, ### [0]
                ).sample 

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                else:
                    assert 0, "Not Implemented"

                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf 
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale) ### TODO optimal noise and LCM

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        # if self.watermark is not None:
        #     image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)









