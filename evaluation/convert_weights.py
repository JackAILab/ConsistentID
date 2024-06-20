import torch
from safetensors import safe_open

### Used to remove redundant parameters and minimize the model.
ckpt = "./pytorch_model.bin"
state_dict = torch.load(ckpt, map_location="cuda")

image_proj_sd = {}
adapter_modules = {}
FacialEncoder = {}

# open('./model_struct.txt', 'w').write('\n'.join([name for name in state_dict.keys()])) ### view model structures

for k in state_dict:
    if k.startswith("unet"):
        pass ### unet freezed
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = state_dict[k]
    elif k.startswith("adapter_modules"):   
        adapter_modules[k.replace("adapter_modules.", "")] = state_dict[k]
    elif k.startswith("FacialEncoder"):
        FacialEncoder[k.replace("FacialEncoder.", "")] = state_dict[k]

state_dict_path = "./ConsistentID-v1.bin"
torch.save({"image_proj_model": image_proj_sd, "adapter_modules": adapter_modules, "FacialEncoder": FacialEncoder}, state_dict_path)
print(f"Sucessful saved at: {state_dict_path}")




