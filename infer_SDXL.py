import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusionXL_ConsistentID import ConsistentIDStableDiffusionXLPipeline
import sys
from PIL import Image
import numpy as np
import argparse

def infer(base_model=None, star_name=None, prompt=None, face_caption=None):
    # import base SD model and pretrained ConsistentID model
    device = "cuda"
    base_model_path = base_model # "SG161222/Realistic_Vision_V6.0_B1_noVAE"
    consistentID_path = "JackAILab/ConsistentID/ConsistentID_SDXL-v1.bin"
    image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    bise_net_cp = "JackAILab/ConsistentID/face_parsing.pth" ### Please specify the specific path.

    ### Load base model
    pipe = ConsistentIDStableDiffusionXLPipeline.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16, 
        safety_checker=None, # use_safetensors=True, 
        variant="fp16"
    ).to(device)

    ### Load consistentID_model checkpoint
    pipe.load_ConsistentID_model(
        os.path.dirname(consistentID_path),
        image_encoder_path=image_encoder_path,
        bise_net_cp=bise_net_cp,
        subfolder="",
        weight_name=os.path.basename(consistentID_path),
        trigger_word="img",
    )     

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    # lora_model_name = os.path.basename(lora_path)
    # pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name) # trigger: HTA
    ### If there's a specific adapter name defined for this LoRA, use it; otherwise, the default might work.
    ### Ensure 'adapter_name' matches what you intend to use or remove if not needed in your setup.
    # pipe.set_adapter_settings(adapter_name="your_adapter_name_here") # Uncomment and adjust as necessary
    # pipe.set_adapters(,["ConsistentID", "more_art-full"] adapter_weights=[1.0, 0.5])
    ### Fuse the loaded LoRA into the pipeline
    # pipe.fuse_lora()

    ### input image
    input_image_path = f"./examples/{star_name}.jpg"
    select_images = load_image(input_image_path)

    # hyper-parameter
    num_steps = 50
    merge_steps = 30
    negative_prompt = '(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth'
    
    generator = torch.Generator(device=device).manual_seed(222)

    images = pipe(
        prompt=prompt,
        width=864,    
        height=1152, ## 1024 896
        input_id_images=select_images,
        input_image_path=input_image_path,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=merge_steps,
        generator=generator,
    ).images[0]

    save_image_path = f"./{base_model}_{star_name}.png"
    if not os.path.exists(os.path.dirname(save_image_path)):
        os.makedirs(os.path.dirname(save_image_path))

    images.save(save_image_path)
    print(f"IMAGE saved at : {save_image_path}")


if __name__ == "__main__":
    
    ### set parameters
    parser = argparse.ArgumentParser(description="Parse image processing paths.")
    parser.add_argument("--base_model", type=str, default="SG161222/Realistic_Vision_V6.0_B1_noVAE",
                        help="Path to the origin images.")
    parser.add_argument("--star_name", type=str, default="scarlett_johansson", ### albert_einstein  scarlett_johansson
                        help="Path to the origin images.") 
    parser.add_argument("--prompt", type=str, default="A woman wearing a santa hat",
                        help="")                                                                        
    args = parser.parse_args()

    infer(model_name=args.model_name, base_model=args.base_model, star_name=args.star_name, prompt=args.prompt, face_caption=face_caption)





