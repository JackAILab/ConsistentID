'''
Official implementation of FGID data (facial mask) production script
Author: Jiehui Huang
Hugging Face Demo: https://huggingface.co/spaces/JackAILab/ConsistentID
Project: https://ssugarwh.github.io/consistentid.github.io/
'''

import os
import json
import torch
import torchvision.transforms as transforms
import cv2
import os.path as osp
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from models.BiSeNet.model import BiSeNet
from PIL import Image

# file_name_length = -5 # .jpeg
file_name_length = -4 # .jpg / .png

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path=None, \
                    color_save_path=None, json_save_path=None):
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

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1): # num_of_class=17 pi=1~16
        index = np.where(vis_parsing_anno == pi) # index[0/1].shape.(11675,) 
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi] # vis_parsing_anno_color.shape.(512, 512, 3)

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(color_save_path[:file_name_length] +'_color.png', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100]) 
        cv2.imwrite(save_path[:file_name_length] +'_mask.png', vis_parsing_anno) 


def evaluate(respth="./parsing_mask_IMG", color_respth="./parsing_color_IMG", resize_dspth="./resize_IMG", \
             dspth='./origin_IMG', jp="./all_JSON", cp='./79999_iter.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

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
    with torch.no_grad():
        total_files = sum(len(files) for _, _, files in os.walk(dspth))
        for root, dirs, files in os.walk(dspth):
            for image_file in tqdm(files, total=total_files, desc="Processing Files"):
                img_path = osp.join(root, image_file)
                img = Image.open(img_path)
                image = img.resize((512, 512), Image.BILINEAR)
                relative_path = osp.relpath(root, dspth)
                resize_dspth_path = osp.join(resize_dspth, relative_path, image_file[:file_name_length] +'_resize.png') # image_file  '0062963.png'
                os.makedirs(os.path.dirname(resize_dspth_path), exist_ok=True)
                image.save(resize_dspth_path) # save origin resized IMGs

                ### Cut image
                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                out = net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

                ### Create storage path
                result_path = osp.join(respth, relative_path, image_file)
                color_result_path = osp.join(color_respth, relative_path, image_file) # relative_path: 15
                if file_name_length==-5:
                    json_path = osp.join(jp, relative_path, image_file.replace("jpeg", "json")) 
                elif file_name_length==-4:
                    json_path = osp.join(jp, relative_path, image_file.replace("jpg", "json")) 
                    json_path = osp.join(jp, relative_path, image_file.replace("png", "json")) # save png or jpg

                ### Create subfolder path
                os.makedirs(osp.dirname(result_path), exist_ok=True)
                os.makedirs(osp.dirname(color_result_path), exist_ok=True)
                os.makedirs(osp.dirname(json_path), exist_ok=True)

                ### Save json files
                json_data = {
                    "origin_IMG": osp.join(relative_path, image_file), # '15/0062963.png'
                    "resize_IMG": osp.join(relative_path, image_file[:file_name_length] +'_resize.png'), # '15/0062963_resize.png'
                    "parsing_color_IMG": osp.join(relative_path, image_file[:file_name_length] +'_color.png'), # '15/0062963_color.png'
                    "parsing_mask_IMG": osp.join(relative_path, image_file[:file_name_length] +'_mask.png'),                            
                }
                with open(json_path, 'w') as json_file:
                    json.dump(json_data, json_file)
                print(f"JSON File Saved Success at {json_file}!")

                ### save origin fine_mask color IMGs
                vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=result_path, color_save_path=color_result_path, json_save_path=json_path)



if __name__ == "__main__":
    
    ### Specify folder path
    origin_img_path = "./FGID_Origin" # read_path for customized IMGs
    pretrain_model_path = "JackAILab/ConsistentID/face_parsing.pth" # read_path for pretrained face-parsing model

    resize_IMG_path = "./FGID_resize" # save_path for customized resized IMGs
    parsing_mask_IM_path = "./FGID_parsing_mask" # save_path for fine_mask gray IMGs
    parsing_color_IM_path = "./FGID_parsing_color" # save_path for fine_mask color IMGs
    json_save_path = "./FGID_JSON" # save_path for fine_mask information

    evaluate(respth=parsing_mask_IM_path, color_respth=parsing_color_IM_path, resize_dspth=resize_IMG_path, \
            dspth=origin_IMG_path, jp=json_save_path, cp=pretrain_model_path)




''' 
Additional instructions:

(1) Specify folder path, then CUDA_VISIBLE_DEVICES=0 python ./FGID_mask.py

(2) Time consumption: 10w IMGs / 15 hours on single RTX3090

(3) JSON data should be constructed as follows:
{
"origin_IMG": "15/0062963.png", 
"resize_IMG": "15/0062963_resize.png", 
"parsing_color_IMG": "15/0062963_color.png", 
"parsing_mask_IMG": "15/0062963_mask.png"
}

(4) Reference table of facial parts corresponding to the Mak value of Face-parsing:

|  Mask  | Face part name         | RGB Color           |
|--------|------------------------|---------------------|
| 1      | Face                   | [255, 0, 0]         |
| 2      | Left_eyebrow           | [255, 85, 0]        |
| 3      | Right_eyebrow          | [255, 170, 0]       |
| 4      | Left_eye               | [255, 0, 85]        |
| 5      | Right_eye              | [255, 0, 170]       |
| 6      | Hair                   | [0, 0, 255]         |
| 7      | Left_ear               | [85, 0, 255]        |
| 8      | Right_ear              | [170, 0, 255]       |
| 9      | Mouth_external_contour | [0, 255, 85]        |
| 10     | Nose                   | [0, 255, 0]         |
| 11     | Mouth_inner_contour    | [0, 255, 170]       |
| 12     | Upper_lip              | [85, 255, 0]        |
| 13     | Lower_lip              | [170, 255, 0]       |
| 14     | Neck                   | [0, 85, 255]        |
| 15     | Neck_inner_contour     | [0, 170, 255]       |
| 16     | Cloth                  | [255, 255, 0]       |
| 17     | Hat                    | [255, 0, 255]       |
| 18     | Earring                | [255, 85, 255]      |
| 19     | Necklace               | [255, 255, 85]      |
| 20     | Glasses                | [255, 170, 255]     |
| 21     | Hand                   | [255, 0, 255]       |
| 22     | Wristband              | [0, 255, 255]       |
| 23     | Clothes_upper          | [85, 255, 255]      |
| 24     | Clothes_lower          | [170, 255, 255]     |
| 25     | Other                  | [0, 0, 0]           |

'''



