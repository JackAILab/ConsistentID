import torch
import os
# from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from diffusers import ControlNetModel, DDIMScheduler
from pipelines.StableDIffusionInpaint_ConsistentID import StableDiffusionInpaintConsistentIDPipeline
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

# Set device and define model paths
# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available. Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device('cpu')
    print("GPU is not available. Using CPU.")
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
consistentID_path = "JackAILab/ConsistentID/ConsistentID-v1.bin"

# Load initial and mask images
# init_image_url = " " # TODO need to be checked
# mask_image_url = " "
init_image = load_image(init_image_url)
mask_image = load_image(mask_image_url)

# Resize images
select_images = init_image.resize((512, 512))
mask_image = mask_image.resize((512, 512))

# Create control image using Canny edge detection
def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

control_image = make_canny_condition(init_image)

# Load control model for inpainting
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", 
    torch_dtype=torch.float16,
).to(device) 

# Load base model
pipe = StableDiffusionInpaintConsistentIDPipeline.from_pretrained(
    base_model_path, 
    controlnet=controlnet, 
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16",
).to(device)

# Load ConsistentID model checkpoint
pipe.load_ConsistentID_model(
    os.path.dirname(consistentID_path),
    subfolder="",
    weight_name=os.path.basename(consistentID_path),
    trigger_word="img",
)

# Set up scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.to(device)

# Set generator with seed
generator = torch.Generator(device=device).manual_seed(2024)
# hyper-parameter
num_steps = 50
merge_steps = 30
# Define prompt and parameters
prompt = "cinematic photo, A man, in a forest, adventuring, 50mm photograph, half-length portrait, film, bokeh, professional, 4k, highly detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))). out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))"

# Generate the image
images = pipe(
    prompt=prompt,
    width=512,    
    height=768,
    strength=1,
    mask_image=mask_image,
    input_id_images=select_images,
    control_image=control_image,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=merge_steps,
    generator=generator,
).images[0]

# Save the resulting image
images.save("./result.jpg")
