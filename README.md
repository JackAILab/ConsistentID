# ConsistentID
Customized ID Consistent for human<br>
[Demo Link](http://consistentid.natapp1.cc/)<br>
[Project page Link]()<br>
[Paper Link](https://ssugarwh.github.io/consistentid.github.io/)

Diffusion-based technologies have made significant strides, particularly in personalized and customized facial generation.  
However, existing methods face challenges in achieving high-fidelity and detailed identity (ID) consistency, primarily due to insufficient fine-grained control over facial areas and the lack of a comprehensive strategy for ID preservation by fully considering intricate facial details and the overall face. 
To address these limitations, we introduce ConsistentID, an innovative method crafted for diverse identity-preserving portrait generation under fine-grained multimodal facial prompts, utilizing only a single reference image.  
ConsistentID comprises two key components: a multimodal facial prompt generator that combines facial features, corresponding facial descriptions and the overall facial context to enhance precision in facial details, and an ID-preservation network optimized through the facial attention localization strategy, aimed at preserving ID consistency in facial regions.  
Together, these components significantly enhance the accuracy of ID preservation by introducing fine-grained multimodal ID information from facial regions.  
To facilitate training of ConsistentID, we present a fine-grained portrait dataset, FGID, with over 500,000 facial images, offering greater diversity and comprehensiveness than existing public facial datasets. % such as LAION-Face, CelebA, FFHQ, and SFHQ. 
Experimental results substantiate that our ConsistentID achieves exceptional precision and diversity in personalized facial generation, surpassing existing methods in the MyStyle dataset.  
Furthermore, while ConsistentID introduces more multimodal ID information, it maintains a fast inference speed during generation. 
Code, models, and datasets are provided in the supplementary materials to enable the reproduction of performance. 



## Requirements

To install requirements:

```setup
pip3 install -r requirements.txt
```

## Data Preparation

Prepare Data in the following format

    ├── data
    |   ├── JSON_all.json 
    |   ├── resize_IMG # Imgaes 
    |   ├── all_faceID  # FaceID
    |   └── parsing_mask_IMG # Parsing Mask 

The .json file should be like
```
[
    {
        "resize_IMG": "Path to resized image...",
        "parsing_color_IMG": "...",
        "parsing_mask_IMG": "...",
        "vqa_llva": "...",
        "id_embed_file_resize": "...",
        "vqa_llva_more_face_detail": "..."
    },
    ...
]
```

## Train

```setup
bash train_bash.sh
```

## Infer

```setup
python infer.py
```

## Model weights

We will upload pretrained weights as soon as possialbe. Feel free to check our model structure for now.
