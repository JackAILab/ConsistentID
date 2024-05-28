'''
Official implementation of FGID data (facial caption) production script
Author: Jiehui Huang
Hugging Face Demo: https://huggingface.co/spaces/JackAILab/ConsistentID
Project: https://ssugarwh.github.io/consistentid.github.io/
'''

import os
import cv2
import torch
import json

from tqdm import tqdm
from insightface.app import FaceAnalysis
### Initialize the FaceAnalysis model
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def process_faceid(origin_path=None, resize_path=None,faceid_path=None, json_path=None):

    ### ========================== resize extraction ==========================
    fail_count = 0
    success_count = 0

    total_files = sum(len(files) for _, _, files in os.walk(resize_path))

    for root, dirs, files in os.walk(resize_path):
        for image_file in tqdm(files, total=total_files, desc="Processing Files"):
            image_path = os.path.join(root, image_file) # '023_Origin.jpeg'
            image = cv2.imread(image_path)
            faces = app.get(image) 
            relative_path = os.path.relpath(root, resize_path) 
        
            image_id = os.path.splitext(image_file)[0] # '023_Origin'
            # Save faceid_embeds
            if faces == []:   
                ### TODO The prior extraction of FaceID is unstable and a stronger ID prior structure can be used.
                fail_count += 1
                print(f"fail_count resize number is {fail_count}, the current fail img is {image_id}")
                continue            
            else:
                faceid_embed_file = os.path.join(faceid_path, relative_path, image_id+'_faceid.bin')
                ### Ensure the directory exists
                os.makedirs(os.path.join(faceid_path, relative_path), exist_ok=True)
                faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                torch.save(faceid_embeds, faceid_embed_file)
                success_count += 1 
                print(f"success_count resize number is {success_count}, the current success img is {os.path.join(relative_path,image_id)}")
            
            ### To update the corresponding JSON file, you need to first run the FGID_mask.py
            json_file_path = os.path.join(json_path, relative_path, image_id.replace('_resize', '')+'.json')

            with open(json_file_path, 'r+') as json_file:
                data = json.load(json_file)
                data['id_embed_file_resize'] = os.path.join(relative_path, image_id+'_faceid.bin')
                json_file.seek(0)
                json.dump(data, json_file)
                json_file.truncate()

            # print(f"Success_count number is {success_count}, Saved faceid_embeds: {faceid_embed_path}, Updated JSON: {json_file_path}")

    ### ========================== origin again ==========================
    fail_count = 0
    success_count = 0

    total_files = sum(len(files) for _, _, files in os.walk(origin_path))

    for root, dirs, files in os.walk(origin_path):
        for image_file in tqdm(files, total=total_files, desc="Processing Files"):
            image_path = os.path.join(root, image_file) # '023_Origin.jpeg'
            image = cv2.imread(image_path)
            faces = app.get(image)
            relative_path = os.path.relpath(root, origin_path)
            
            image_id = os.path.splitext(image_file)[0] # '023_Origin'
            if faces == []:
                fail_count += 1
                continue            
            else:
                faceid_embed_file = os.path.join(faceid_path, relative_path, image_id+'_faceid.bin')
                ### Ensure the directory exists
                os.makedirs(os.path.join(faceid_path, relative_path), exist_ok=True)
                faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                torch.save(faceid_embeds, faceid_embed_file)
                success_count += 1 
                print(f"success_count origin number is {success_count}, the current success img is {os.path.join(relative_path,image_id)}")
            
            json_file_path = os.path.join(json_path, relative_path, image_id+'.json')
            with open(json_file_path, 'r+') as json_file:
                data = json.load(json_file)
                data['id_embed_file_origin'] = os.path.join(relative_path, image_id+'_faceid.bin')
                json_file.seek(0) 
                json.dump(data, json_file)
                json_file.truncate()

            # print(f"Success_count number is {success_count}, Saved faceid_embeds: {faceid_embed_path}, Updated JSON: {json_file_path}")

if __name__ == "__main__":
    
    ### Specify folder path
    origin_img_path = "./FGID_Origin" # read_path for customized IMGs
    resize_IMG_path = "./FGID_resize" # read_path for customized resized IMGs (generated in FGID_mask.py)

    faceid_path = "./FGID_faceID" # save_path for faceid embedding (.bin)
    json_save_path = "./FGID_JSON" # save_path for faceid information

    process_faceid(origin_path=origin_img_path, resize_path=resize_IMG_path, \
                    faceid_path=faceid_path, json_path=json_save_path)

'''
Additional instructions:

(1) Specify folder path, then CUDA_VISIBLE_DEVICES=0 python ./FGID_faceid_embeds.py

(2) Note: Run FGID_mask.py first, and then FGID_faceid_embeds.py to ensure that the json data is created smoothly.

(3) JSON data should be constructed as follows:
{
"origin_IMG": "15/0062963.png", 
"resize_IMG": "15/0062963_resize.png", 
"parsing_color_IMG": "15/0062963_color.png", 
"parsing_mask_IMG": "15/0062963_mask.png"
"id_embed_file_resize": "15/0061175_resize_faceid.bin", ### may not exist, and the training process uses ZERO embedding instead.
"id_embed_file_origin": "15/0061175_faceid.bin", ### may not exist
"vqa_llva": "The image features a young woman with short hair, wearing a black shirt and a black jacket. 
            She has a smiling expression on her face, and her eyes are open. The woman appears to be the main subject of the photo.",
"vqa_llva_face_caption": "The person in the image has a short haircut, which gives her a modern and stylish appearance. 
            She has a small nose, which is a prominent feature of her face. Her eyes are large and wide-set, adding to her striking facial features. 
            The woman has a thin mouth and a small chin, which further accentuates her overall look. Her ears are small and subtle, blending in with her hairstyle."
}
'''






