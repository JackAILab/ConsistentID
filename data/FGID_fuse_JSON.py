'''
Official implementation of FGID data (facial caption) production script
Author: Jiehui Huang
Hugging Face Demo: https://huggingface.co/spaces/JackAILab/ConsistentID
Project: https://ssugarwh.github.io/consistentid.github.io/
'''

import os
import json

from tqdm import tqdm

json_folder_path = "./FGID_JSON" # read_path for all FGID json data
output_file_path = "./JSON_all.json" # save_path for all FGID json data

all_data = []


total_files = sum(len(files) for _, _, files in os.walk(json_folder_path))
for root, dirs, files in os.walk(json_folder_path):
    for file_name in tqdm(files, total=total_files, desc="Processing Files"): # (('.png', '.jpg', '.jpeg')):
        json_path = os.path.join(root, file_name)
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
            all_data.append(json_data)

### Write all data to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(all_data, output_file)

print(f"Merged all JSON files into {output_file_path}")





