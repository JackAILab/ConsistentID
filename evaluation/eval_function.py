from PIL import Image
import numpy as np
import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
import random
from copy import deepcopy

def crop_image(image_pil, threshold=10):

    # image = Image.open(image_path)
    image = image_pil
    gray_image = image.convert('L')
    gray_array = np.array(gray_image)

    # find black
    non_black_indices = np.where(gray_array > threshold)
    top = np.min(non_black_indices[0])
    bottom = np.max(non_black_indices[0])
    left = np.min(non_black_indices[1])
    right = np.max(non_black_indices[1])

    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image


def prepare_image_token_idx(image_token_mask, max_num_objects):

    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat(
            [ 
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [ 
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )
    
    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    
    return image_token_idx, image_token_idx_mask


def white_balance_correction(image):

    image_float = image.astype(np.float64)

    mean_r = np.mean(image_float[:, :, 0])
    mean_g = np.mean(image_float[:, :, 1])
    mean_b = np.mean(image_float[:, :, 2])

    offset_r = 128 - mean_r
    offset_g = 128 - mean_g
    offset_b = 128 - mean_b

    image_balanced = image_float + [offset_r, offset_g, offset_b]

    image_balanced = np.clip(image_balanced, 0, 255).astype(np.uint8)

    return image_balanced
 
def get_object_transforms(args):
    if args.no_object_augmentation:
        pre_augmentations = []
        augmentations = []
    else:
        pre_augmentations = [
            (
                "zoomin",
                T.RandomApply([RandomZoomIn(min_zoom=1.0, max_zoom=2.0)], p=0.5),
            ),
        ]

        augmentations = [
            (
                "rotate",
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=30, interpolation=T.InterpolationMode.BILINEAR
                        )
                    ],
                    p=0.75,
                ),
            ),
            ("jitter", T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.5)),
            ("blur", T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)),
            ("gray", T.RandomGrayscale(p=0.1)),
            ("flip", T.RandomHorizontalFlip()),
            ("elastic", T.RandomApply([T.ElasticTransform()], p=0.5)),
        ]

    object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                *pre_augmentations,
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (args.resolution, args.resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ),
                *augmentations,
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    return object_transforms
    
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from collections import OrderedDict

class PadToSquare(torch.nn.Module):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h == w:
            return image
        elif h > w:
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                self.padding_mode,
                self.fill,
            )
        else:
            padding = (w - h) // 2
            image = torch.nn.functional.pad(
                image,
                (0, 0, padding, padding),
                self.padding_mode,
                self.fill,
            )
        return image

class CropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h <= w:
            return image
        return image[:, :w, :]

class RandomZoomIn(torch.nn.Module):
    def __init__(self, min_zoom=1.0, max_zoom=1.5):
        super().__init__()
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def forward(self, image: torch.Tensor):
        zoom = torch.rand(1) * (self.max_zoom - self.min_zoom) + self.min_zoom
        original_shape = image.shape
        image = T.functional.resize(
            image,
            (int(zoom * image.shape[1]), int(zoom * image.shape[2])),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        image = CropTopSquare()(image)
        return image

import sys
sys.path.append("/mnt/data/sysu/Users/huangjiehui/projects/ConsistentID/")
from models.BiSeNet.model import BiSeNet
import json
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

import torch
import numpy as np
from PIL import Image

def resize_tensor(input_tensor, size):

    numpy_image = np.transpose(input_tensor.cpu().numpy(), (1, 2, 0))

    pil_image = Image.fromarray(np.uint8(numpy_image))

    resized_image = pil_image.resize(size, Image.BILINEAR)

    resized_numpy_image = np.array(resized_image)
    resized_numpy_image = np.transpose(resized_numpy_image, (2, 0, 1))

    resized_image_tensor = torch.tensor(resized_numpy_image, dtype=input_tensor.dtype)

    return resized_image_tensor

def parsing_face_mask(raw_image_refer, cp='/mnt/data/sysu/Users/huangjiehui/pretrained_model/ConsistentID/face_parsing.pth'):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = cp
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    to_pil = transforms.ToPILImage()

    with torch.no_grad():

        image = raw_image_refer.resize((512, 512), Image.BILINEAR)
        image_resize_PIL = image
        img = to_tensor(image)

        img = torch.unsqueeze(img, 0)
        img = img.float().cuda()
        out = net(img)[0]
        parsing_anno = out.squeeze(0).cpu().numpy().argmax(0)

    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    im = np.array(image_resize_PIL)
    vis_im = im.copy().astype(np.uint8)
    stride=1
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_parsing_anno_color = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_parsing_anno_color, vis_parsing_anno

def fetch_mask_raw_image(raw_image, mask_image):
    
    mask_image = mask_image.resize(raw_image.size)
    mask_raw_image = Image.composite(raw_image, Image.new('RGB', raw_image.size, (0, 0, 0)), mask_image) 

    return mask_raw_image

mapping_table = [
    {"Mask Value": 0, "Body Part": "Background", "RGB Color": [0, 0, 0]},
    {"Mask Value": 1, "Body Part": "Face", "RGB Color": [255, 0, 0]},
    {"Mask Value": 2, "Body Part": "Left_Eyebrow", "RGB Color": [255, 85, 0]},
    {"Mask Value": 3, "Body Part": "Right_Eyebrow", "RGB Color": [255, 170, 0]},
    {"Mask Value": 4, "Body Part": "Left_Eye", "RGB Color": [255, 0, 85]},
    {"Mask Value": 5, "Body Part": "Right_Eye", "RGB Color": [255, 0, 170]},
    {"Mask Value": 6, "Body Part": "Hair", "RGB Color": [0, 0, 255]},
    {"Mask Value": 7, "Body Part": "Left_Ear", "RGB Color": [85, 0, 255]},
    {"Mask Value": 8, "Body Part": "Right_Ear", "RGB Color": [170, 0, 255]},
    {"Mask Value": 9, "Body Part": "Mouth_External Contour", "RGB Color": [0, 255, 85]},
    {"Mask Value": 10, "Body Part": "Nose", "RGB Color": [0, 255, 0]},
    {"Mask Value": 11, "Body Part": "Mouth_Inner_Contour", "RGB Color": [0, 255, 170]},
    {"Mask Value": 12, "Body Part": "Upper_Lip", "RGB Color": [85, 255, 0]}, 
    {"Mask Value": 13, "Body Part": "Lower_Lip", "RGB Color": [170, 255, 0]},
    {"Mask Value": 14, "Body Part": "Neck", "RGB Color": [0, 85, 255]},
    {"Mask Value": 15, "Body Part": "Neck_Inner Contour", "RGB Color": [0, 170, 255]},
    {"Mask Value": 16, "Body Part": "Cloth", "RGB Color": [255, 255, 0]},
    {"Mask Value": 17, "Body Part": "Hat", "RGB Color": [255, 0, 255]},
    {"Mask Value": 18, "Body Part": "Earring", "RGB Color": [255, 85, 255]},
    {"Mask Value": 19, "Body Part": "Necklace", "RGB Color": [255, 255, 85]},
    {"Mask Value": 20, "Body Part": "Glasses", "RGB Color": [255, 170, 255]},
    {"Mask Value": 21, "Body Part": "Hand", "RGB Color": [255, 0, 255]},
    {"Mask Value": 22, "Body Part": "Wristband", "RGB Color": [0, 255, 255]},
    {"Mask Value": 23, "Body Part": "Clothes_Upper", "RGB Color": [85, 255, 255]},
    {"Mask Value": 24, "Body Part": "Clothes_Lower", "RGB Color": [170, 255, 255]}
]

def masks_for_unique_values(image_raw_mask):

    image_array = np.array(image_raw_mask)
    unique_values, counts = np.unique(image_array, return_counts=True)
    masks_dict = {}

    for value in unique_values:
        binary_image = np.uint8(image_array == value) * 255
        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(image_array)

        for contour in contours:
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        if value == 0:
            body_part="WithoutBackground"
            mask2 = np.where(mask == 255, 0, 255).astype(mask.dtype)
            masks_dict[body_part] = Image.fromarray(mask2)

        body_part = next((entry["Body Part"] for entry in mapping_table if entry["Mask Value"] == value), f"Unknown_{value}")

        if body_part.startswith("Unknown_"):
            continue            

        masks_dict[body_part] = Image.fromarray(mask)
    
    return masks_dict

import torch
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers import PretrainedConfig

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

import re
def remove_duplicate_keywords(text, keywords):
    keyword_counts = {}

    words = re.findall(r'\b\w+\b|[.,;!?]', text)

    for keyword in keywords:
        keyword_counts[keyword] = 0

        for i, word in enumerate(words):
            if word.lower() == keyword.lower():
                keyword_counts[keyword] += 1

                if keyword_counts[keyword] > 1:
                    words[i] = ""

    processed_text = " ".join(words)

    return processed_text

def process_text_with_markers(text, parsing_mask_list):

    keywords = ["face", "eyes", "ears", "nose", "mouth"]

    text = remove_duplicate_keywords(text, keywords)

    key_parsing_mask_markers = ["Face", "Left_Eye", "Right_Eye", "Left_Ear", "Right_Ear", "Nose", "Upper_Lip", "Lower_Lip"]
    mapping = {
        "Face": "face",
        "Left_Eye": "eyes",
        "Right_Eye": "eyes",
        "Left_Ear": "ears",
        "Right_Ear": "ears",        
        "Nose": "nose",
        "Upper_Lip": "mouth",
        "Lower_Lip": "mouth",
    }
    facial_features_align = []
    markers_align = []
    for key in key_parsing_mask_markers:
        if key in parsing_mask_list:
            mapped_key = mapping.get(key, key.lower())
            if mapped_key not in facial_features_align:
                facial_features_align.append(mapped_key)
                markers_align.append("<|"+mapped_key+"|>")

    # (2)
    text_marked = text
    align_parsing_mask_list = parsing_mask_list
    for feature, marker in zip(facial_features_align[::-1], markers_align[::-1]):
        pattern = rf'\b{feature}\b' # feature 就是 "face", "ears", "nose", "eyes", "mouth" 
    
        text_marked_new = re.sub(pattern, f'{feature} {marker}', text_marked, count=1)
        if text_marked == text_marked_new:
            for key, value in mapping.items():
                if value == feature:
                    if key in align_parsing_mask_list:
                        del align_parsing_mask_list[key]   

        text_marked = text_marked_new 

    text_marked = text_marked.replace('\n', '')

    # (3)
    ordered_text = []
    text_none_makers = []
    facial_marked_count = 0 
    skip_count = 0
    for marker in markers_align: # markers_align ['<|face|>', '<|eyes|>', '<|nose|>', '<|mouth|>']
        start_idx = text_marked.find(marker)
        end_idx = start_idx + len(marker)

        while start_idx > 0 and text_marked[start_idx - 1] not in [",", ".", ";"]: # [",", ".", ";"] [ "." ]
            start_idx -= 1

        while end_idx < len(text_marked) and text_marked[end_idx] not in [",", ".", ";"]: # [",", ".", ";"] [ "." ]
            end_idx += 1

        context = text_marked[start_idx:end_idx].strip()
        if context == "":
            text_none_makers.append(text_marked[:end_idx])
            # print(f"The facial part of {marker} can not be found in caption!\r\n")
        else:
            if skip_count!=0:
                skip_count -= 1 
                continue
            else:
                ordered_text.append(context + ",")
                text_delete_makers = text_marked[:start_idx] + text_marked[end_idx:]
                text_marked = text_delete_makers
                facial_marked_count += 1
                # print(f"Current successful matched special token in the facial text number is: {facial_marked_count}")
                # print(f"The current marked text is: {ordered_text} \r\n")

    align_marked_text = " ".join(ordered_text)
    replace_list = ["<|face|>", "<|eyes|>", "<|ears|>", "<|nose|>", "<|mouth|>"] 
    for item in replace_list:
        align_marked_text = align_marked_text.replace(item, "<|facial|>")

    # print(f"The final aligned facial text is: {align_marked_text}")

    return align_marked_text, align_parsing_mask_list


def tokenize_and_mask_noun_phrases_ends(text, image_token_id, facial_token_id, tokenizer):

    input_ids = tokenizer.encode(text)
    image_noun_phrase_end_mask = [False for _ in input_ids] 
    facial_noun_phrase_end_mask = [False for _ in input_ids]
    clean_input_ids = []
    clean_index = 0
    image_num = 0

    for i, id in enumerate(input_ids):
        if id == image_token_id:
            image_noun_phrase_end_mask[clean_index + image_num - 1] = True
            image_num += 1
        elif id == facial_token_id:
            facial_noun_phrase_end_mask[clean_index - 1] = True        
        else:
            clean_input_ids.append(id)
            clean_index += 1

    max_len = tokenizer.model_max_length 

    if len(clean_input_ids) > max_len:
        clean_input_ids = clean_input_ids[:max_len]
    else:
        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
            max_len - len(clean_input_ids)
        )

    if len(image_noun_phrase_end_mask) > max_len: 
        image_noun_phrase_end_mask = image_noun_phrase_end_mask[:max_len]
    else:
        image_noun_phrase_end_mask = image_noun_phrase_end_mask + [False] * (
            max_len - len(image_noun_phrase_end_mask)
        )

    if len(facial_noun_phrase_end_mask) > max_len: 
        facial_noun_phrase_end_mask = facial_noun_phrase_end_mask[:max_len]
    else:
        facial_noun_phrase_end_mask = facial_noun_phrase_end_mask + [False] * (
            max_len - len(facial_noun_phrase_end_mask)
        )        

    clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
    image_noun_phrase_end_mask = torch.tensor(image_noun_phrase_end_mask, dtype=torch.bool)
    facial_noun_phrase_end_mask = torch.tensor(facial_noun_phrase_end_mask, dtype=torch.bool)

    return clean_input_ids.unsqueeze(0), image_noun_phrase_end_mask.unsqueeze(0), facial_noun_phrase_end_mask.unsqueeze(0)

def prepare_image_token_idx(image_token_mask, facial_token_mask, max_num_objects=2, max_num_facials=5):

    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects: 
        image_token_idx = torch.cat(
            [ 
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [ 
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )

    facial_token_idx = torch.nonzero(facial_token_mask, as_tuple=True)[1]
    facial_token_idx_mask = torch.ones_like(facial_token_idx, dtype=torch.bool)     
    if len(facial_token_idx) < max_num_facials:
        facial_token_idx = torch.cat(
            [ 
                facial_token_idx,
                torch.zeros(max_num_facials - len(facial_token_idx), dtype=torch.long),
            ]
        )
        facial_token_idx_mask = torch.cat(
            [ 
                facial_token_idx_mask,
                torch.zeros(
                    max_num_facials - len(facial_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )

    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    
    facial_token_idx = facial_token_idx.unsqueeze(0)
    facial_token_idx_mask = facial_token_idx_mask.unsqueeze(0)

    return image_token_idx, image_token_idx_mask, facial_token_idx, facial_token_idx_mask 

class Transform_ID():
    def __init__(self,):
        super().__init__()
        pass
    
    def process_image(self, image):

        if image.size(0) > 3:
            image_list = [image]
            processed_image = torch.cat(image_list, dim=0)
        else:
            image_list = [image] * 2

        processed_image = torch.cat(image_list, dim=0)

        return processed_image

    def process_clip_image(self, clip_image):

        if clip_image.size(0) > 1:
            processed_clip_image = torch.cat([clip_image, clip_image], dim=0)
        else:
            processed_clip_image = clip_image

        return processed_clip_image

    def process_pasring_mask(self, parsing_mask):

        processed_parsing_mask=[]

        return processed_parsing_mask

    def process_faceid(self, faceid):

        processed_faceid=[]   

        return processed_faceid

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)    

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return updated_prompt_embeds


if __name__ == "__main__":
    PhotoMakerIDEncoder()