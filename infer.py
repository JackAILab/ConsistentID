import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline
import sys


### Download from huggingface, then put the model local, then place the model in a local directory and specify the directory location.
device = "cuda"
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
consistentID_path = "JackAILab/ConsistentID/ConsistentID-v1.bin" 

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

### Experimental feature, using LoRA modules in community
# lora_model_name = os.path.basename(lora_path)
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name) # trigger: HTA
### If there's a specific adapter name defined for this LoRA, use it; otherwise, the default might work.
### Ensure 'adapter_name' matches what you intend to use or remove if not needed in your setup.
# pipe.set_adapter_settings(adapter_name="your_adapter_name_here") # Uncomment and adjust as necessary
# pipe.set_adapters(,["ConsistentID", "more_art-full"] adapter_weights=[1.0, 0.5]) # TODO
### Fuse the loaded LoRA into the pipeline
# pipe.fuse_lora()

### input image 
select_images = load_image(script_directory+"/images/person.jpg")
# hyper-parameter
num_steps = 50
merge_steps = 30
# Prompt
prompt = "A man, in a forest, adventuring"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

### Extend Prompt
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

images.save("./images/result.jpg")


