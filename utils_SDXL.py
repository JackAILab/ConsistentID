import argparse
import os
import json
import random
import torch
from PIL import Image
from transformers import CLIPImageProcessor
from torchvision import transforms
from functions_SDXL import crop_image, extract_first_sentence, process_text_with_markers, masks_for_unique_values, fetch_mask_raw_image, tokenize_and_mask_noun_phrases_ends, prepare_image_token_idx
import numpy as np
import pdb

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, json_mutiID_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, \
                image_root_path="", faceid_root_path="", parsing_root_path="", image_token="<|image|>", facial_token="<|facial|>",
                image_token_face="<|face|>",image_token_ear="<|ear|>",image_token_nose="<|nose|>",image_token_eye="<|eye|>",image_token_mouth="<|mouth|>"):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        self.image_root_path = image_root_path
        self.faceid_root_path = faceid_root_path
        self.parsing_root_path = parsing_root_path

        self.data = json.load(open(json_file))
                                   
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.transform_mask = transforms.Compose([
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])        
        self.clip_image_processor = CLIPImageProcessor()
        self.image_token = image_token
        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.facial_token = facial_token
        tokenizer.add_tokens([facial_token], special_tokens=True)
        self.facial_token_id = tokenizer.convert_tokens_to_ids(facial_token)
        self.max_num_facials = 5 
        json_name = json_file.split("/")[-2]
        self.text_path = f"/mnt/data/sysu/Users/huangjiehui/projects/ConsistentID/SDXL_train/results/logs/train_log_{json_name}.txt"
        with open(self.text_path, 'w') as f:
            f.write(f'Training following data: {json_file}\n')

        
    def __getitem__(self, idx):

        item = self.data[idx]

        try:
            text_origin = item["vqa_llva_more_face_detail"]
            image_file = item["origin_IMG"]
            parsing_mask = item["parsing_mask_IMG"]

            image_raw_mask = Image.open(os.path.join(self.parsing_root_path, parsing_mask)) 
            parsing_mask_list = masks_for_unique_values(image_raw_mask) 
            if "id_embed_file_origin" in item:
                faceid_file = item["id_embed_file_origin"]
            else:
                faceid_file = None
                
            # level 2
            raw_image = Image.open(os.path.join(self.image_root_path, image_file)) 
            # level 1
            if faceid_file==None or faceid_file=="":
                face_id_embed = torch.zeros_like(torch.empty((1, 512)))
                with open(self.text_path, 'a') as f:
                    f.write(str(item["resize_IMG"]) + '\n')
                    f.write("This IMG do not have faceid\n")         
            else:
                face_id_embed = torch.load(os.path.join(self.faceid_root_path, faceid_file))
        except Exception as e:
            image_file = "00000/49250326_resize.png"
            raw_image = Image.open("./49250326_resize.png") 
            item["vqa_llva"] = "The image features a woman sitting in a chair, wearing a black dress and a green top. She is holding a microphone and appears to be smiling. The woman is the main subject of the image, and her facial expression and attire suggest that she might be a performer or an interviewer."
            text_origin = "The woman in the image has a beautiful face with a prominent nose, large eyes, and a small mouth. She has long, curly hair that falls down her back. Her ears are positioned on the sides of her head, and she is wearing a black shirt."
            face_id_embed = torch.load("./49250326_faceid.bin")
            image_raw_mask = Image.open("./49250326_mask.png") 
            parsing_mask_list = masks_for_unique_values(image_raw_mask) 

            with open(self.text_path, 'a') as f:
                error_message = f"Error: {type(e).__name__}, Message: {str(e)}"
                try:
                    f.write(str(item['origin_IMG']) + '\n')
                    f.write(f"{error_message}\n")
                except:
                    f.write(f"{error_message}\n")

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        # random crop
        image_tensor = self.transform(raw_image.convert("RGB"))            
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 
        raw_image = raw_image.resize((self.size, self.size))

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

        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        try:
            body_raw_image = fetch_mask_raw_image(raw_image,parsing_mask_list["WithoutBackground"])
        except:
            body_raw_image = raw_image
            with open(self.text_path, 'a') as f:
                f.write(str(image_file) + '\n')             
            print(f"Fail to extract body part name is:{image_file}!")

        body_image = self.transform(body_raw_image.convert("RGB"))
        body_clip_image = self.clip_image_processor(images=body_raw_image, return_tensors="pt").pixel_values 
        multi_image = torch.cat([image, body_image], dim=0)
        multi_clip_image = torch.cat([clip_image, body_clip_image], dim=1)

        max_text_length=340
        text_face, key_parsing_mask_list = process_text_with_markers(text_origin, key_parsing_mask_list)
        text = "Caption:" + extract_first_sentence(item["vqa_llva"]) + " Detail:" + text_face + item["vqa_llva"][len(extract_first_sentence(item["vqa_llva"])):-1]

        if len(self.tokenizer(text,max_length=self.tokenizer.model_max_length, padding="max_length",truncation=False,return_tensors="pt").input_ids[0])!=77:
            text = "Detail:" + text_face + " Caption:" + item["vqa_llva"]

        if len(text_face)>max_text_length:
            text = item["vqa_llva"]
            text_path = "/mnt/data/sysu/Users/huangjiehui/projects/ConsistentID/SDXL_train/results/logs/train_mask_log_v2.txt"
            with open(text_path, 'a') as f:
                f.write(str(text_face) + '\n')                       

        drop_image_embed = 0

        prob = random.random()
        if prob < 0.1:
            text = ""
            multi_clip_image=torch.zeros_like(multi_clip_image)  
            clip_image=torch.zeros_like(clip_image) 
        elif prob < 0.1 + 0:
            multi_clip_image=torch.zeros_like(multi_clip_image)  
            clip_image=torch.zeros_like(clip_image)
        else:
            pass

        text_input_id_all = self.tokenizer(
            text.replace("<|facial|>",""),
            max_length=self.tokenizer.model_max_length, # 77
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_id_all2 = self.tokenizer_2(
            text.replace("<|facial|>",""),
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        # 4.2 ================================= image_token_idx image_token_idx_mask =================================
        clean_input_id, image_token_mask, facial_token_mask = tokenize_and_mask_noun_phrases_ends( # input_ids就是padding后的test_caption ---- image_token_mask 包含了特殊触发词 
            text, self.image_token_id, self.facial_token_id, self.tokenizer
        )

        max_num_objects=2
        max_num_facials=5
        image_token_idx, image_token_idx_mask, facial_token_idx, facial_token_idx_mask = prepare_image_token_idx(
            image_token_mask, facial_token_mask, max_num_objects, max_num_facials # self.max_num_facials
        )
        
        facial_mask = []
        facial_clip_image = []
        num_facial_part = len(key_parsing_mask_list)
        for key in key_parsing_mask_list:
            key_mask=key_parsing_mask_list[key]
            facial_mask.append(self.transform_mask(key_mask))
            key_mask_raw_image = fetch_mask_raw_image(raw_image,key_mask)
           
            parsing_clip_image = self.clip_image_processor(images=key_mask_raw_image, return_tensors="pt").pixel_values
            facial_clip_image.append(parsing_clip_image)            

        padding_ficial_clip_image = torch.zeros_like(torch.zeros([1, 3, 224, 224]))
        padding_ficial_mask = torch.zeros_like(torch.zeros([1, self.size, self.size]))
        
        if num_facial_part < self.max_num_facials:
            facial_clip_image += [torch.zeros_like(padding_ficial_clip_image) for _ in range(self.max_num_facials - num_facial_part) ]
            facial_mask += [ torch.zeros_like(padding_ficial_mask) for _ in range(self.max_num_facials - num_facial_part)]

        facial_clip_image = torch.stack(facial_clip_image, dim=1).squeeze(0)
        facial_mask = torch.stack(facial_mask, dim=0).squeeze(dim=1)

        return {
            "image": image,
            "multi_image": multi_image,
            "facial_clip_image": facial_clip_image,
            "facial_mask": facial_mask,
            "clean_input_id": clean_input_id,
            "text_input_id_all": text_input_id_all,
            "text_input_id_all2": text_input_id_all2,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "image_token_mask": image_token_mask,
            "facial_token_idx": facial_token_idx, 
            "facial_token_idx_mask": facial_token_idx_mask,
            "facial_token_mask": facial_token_mask,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "text_prompt": text,
            "text_origin": text_origin,
            "clip_image": clip_image,
            "multi_clip_image": multi_clip_image,
            "face_id_embed": face_id_embed,
            "drop_image_embed": drop_image_embed,
            "parsing_mask_list": parsing_mask_list,
            "key_parsing_mask_list": key_parsing_mask_list
        }

    def __len__(self):
        return len(self.data)

def collate_fn(data):

    images = torch.stack([example["image"] for example in data])
    multi_images = torch.stack([example["multi_image"] for example in data])
    facial_clip_images = torch.stack([example["facial_clip_image"] for example in data])
    facial_masks = torch.stack([example["facial_mask"] for example in data])

    clean_input_ids = torch.cat([example["clean_input_id"] for example in data], dim=0)
    text_input_id_alls = torch.cat([example["text_input_id_all"] for example in data], dim=0)
    text_input_id_all2s = torch.cat([example["text_input_id_all2"] for example in data], dim=0)
    image_token_idxs = torch.cat([example["image_token_idx"] for example in data], dim=0) 
    image_token_idx_masks = torch.cat([example["image_token_idx_mask"] for example in data], dim=0)  
    image_token_masks = torch.cat([example["image_token_mask"] for example in data], dim=0)
    facial_token_masks = torch.cat([example["facial_token_mask"] for example in data], dim=0)
    facial_token_idxs = torch.cat([example["facial_token_idx"] for example in data], dim=0)
    facial_token_idx_masks = torch.cat([example["facial_token_idx_mask"] for example in data], dim=0)   

    original_sizes = torch.stack([example["original_size"] for example in data])
    crop_coords_top_lefts = torch.stack([example["crop_coords_top_left"] for example in data])
    target_sizes = torch.stack([example["target_size"] for example in data])
    text_prompts = [example["text_prompt"] for example in data]
    text_origins = [example["text_origin"] for example in data]
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    multi_clip_images = torch.cat([example["multi_clip_image"] for example in data], dim=0) 
    face_id_embeds = torch.stack([example["face_id_embed"] for example in data])
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    parsing_mask_lists = [example["parsing_mask_list"] for example in data]
    key_parsing_mask_lists = [example["key_parsing_mask_list"] for example in data]

    return {
        "images": images,
        "multi_images": multi_images,
        "facial_clip_images": facial_clip_images,
        "facial_masks": facial_masks,
        "clean_input_ids": clean_input_ids,
        "text_input_id_alls": text_input_id_alls,
        "text_input_id_all2s": text_input_id_all2s,
        "image_token_idxs": image_token_idxs, 
        "image_token_idx_masks": image_token_idx_masks,  
        "image_token_masks": image_token_masks,
        "facial_token_masks": facial_token_masks,
        "facial_token_idxs": facial_token_idxs, 
        "facial_token_idx_masks": facial_token_idx_masks,  
        "original_sizes": original_sizes,
        "crop_coords_top_lefts": crop_coords_top_lefts,
        "target_sizes": target_sizes,
        "text_prompts": text_prompts,
        "text_origins": text_origins, 
        "clip_images": clip_images,
        "multi_clip_images": multi_clip_images,
        "face_id_embeds": face_id_embeds,   
        "drop_image_embeds": drop_image_embeds,
        "parsing_mask_lists": parsing_mask_lists,
        "key_parsing_mask_lists": key_parsing_mask_lists
    }
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./stable-diffusion-xl-base-1.0",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=4,
        help="Number of tokens to query from the CLIP image encoding.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="./FGID/JSON_train.json",
        required=False,
        help="Training data",
    )
    parser.add_argument(
        "--data_json_mutiID_file",
        type=str,
        default="./FGID/MultiID/JSON_mutiID_all.json",
        required=False,
        help="Training data",
    )        
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="./FGID/ffhq_celebA/origin_IMG",
        required=False,
        help="Training data root path",
    )
    parser.add_argument(
        "--faceid_root_path",
        type=str,
        default="./FGID/ffhq_celebA/all_faceID",
        required=False,
        help="Training data root path",
    )
    parser.add_argument(
        "--parsing_root_path",
        type=str,
        default="./FGID/ffhq_celebA/parsing_mask_IMG",
        required=False,
        help="Training data root path",
    )        
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="/mnt/data/sysu/Users/huangjiehui/pretrained_model/CLIP-ViT-H-14-laion2B-s32B-b79K",
        required=False,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--image_encoder2_path",
        type=str,
        default="/mnt/data/sysu/Users/huangjiehui/pretrained_model/clip-vit-large-patch14",
        required=False,
        help="Path to CLIP image encoder",
    )    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data/sysu/Users/huangjiehui/projects/ConsistentID/SDXL_train/results/outputs/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="/mnt/data/sysu/Users/huangjiehui/projects/ConsistentID/SDXL_train/results/logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mask_loss_prob",
        type=float,
        default=0.5,
    )       
    parser.add_argument(
        "--facial_weight", # facial_weight attn_reg_weight
        type=float,
        default=0.01,
    )           
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--localization_layers", type=int, default=5) # Add from fastcomposer
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100, # 2000
        help=(
            "Save a checkpoint of the training state every X updates"  
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp8", "fp16", "bf16", "fp32"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--train_text_encoder",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--train_image_encoder",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--image_encoder_trainable_layers",
        type=int,
        default=2,
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args













