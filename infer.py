import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline
import sys


# TODO import base SD model and pretrained ConsistentID model
device = "cuda"
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
consistentID_path = "./ConsistentID_model_facemask_pretrain_50w" # pretrained ConsistentID model

# Gets the absolute path of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

### Load base model
pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
).to(device)

### Load consistentID_model checkpoint
pipe.load_ConsistentID_model(
    os.path.dirname(consistentID_path),
    subfolder="",
    weight_name=os.path.basename(consistentID_path),
    trigger_word="img",
)     

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


### input image TODO
select_images = load_image(script_directory+"/images/person.jpg")
# hyper-parameter
num_steps = 50
merge_steps = 30
# Prompt
prompt = "A man, in a forest, adventuring"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

#Extend Prompt
prompt = "cinematic photo," + prompt + ", 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"
negtive_prompt_group="((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
negative_prompt = negative_prompt + negtive_prompt_group

generator = torch.Generator(device=device).manual_seed(2024)

images = pipe(
    prompt=prompt,
    width=512,    
    height=768,
    input_id_images=select_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=merge_steps,
    generator=generator,
).images[0]

images.save(script_directory+"/images/sample.jpg")


