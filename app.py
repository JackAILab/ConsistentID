import gradio as gr
import torch
import os
import glob
from datetime import datetime
from PIL import Image
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline
import sys
# print(gr.__version__)
# 4.16.0

# Gets the absolute path of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

def process(inputImage,prompt,negative_prompt):

    device = "cuda"
    # TODO import base SD model and pretrained ConsistentID model
    base_model_path = ""
    consistentID_path = ""

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
    
    # hyper-parameter
    select_images = load_image(Image.fromarray(inputImage))
    num_steps = 50
    merge_steps = 30
    

    if prompt == "":
        prompt = "A man, in a forest, adventuring"

    if negative_prompt == "":
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    #Extend Prompt
    prompt = "cinematic photo," + prompt + ", 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"

    negtive_prompt_group="((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"
    negative_prompt = negative_prompt + negtive_prompt_group
    
    seed = torch.randint(0, 1000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        prompt=prompt,
        width=512,    
        height=512,
        input_id_images=select_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=merge_steps,
        generator=generator,
    ).images[0]

    current_date = datetime.today()
    images.save(script_directory+f"/images/gradio_outputs/{current_date}-{seed}"+".jpg")
    return script_directory+f"/images/gradio_outputs/{current_date}-{seed}"+".jpg"


iface = gr.Interface(
    fn=process,
    inputs=[
        gr.Image(label="Upload Image"), 
        gr.Textbox(label="prompt",placeholder="A man, in a forest, adventuring"),
        gr.Textbox(label="negative prompt",placeholder="monochrome, lowres, bad anatomy, worst quality, low quality, blurry"),
    ],
    outputs=[
        gr.Image(label="Output"), 
    ],
    title="ConsistentID Demo",
    description="Put reference portrait below"
)

iface.launch(share=True)