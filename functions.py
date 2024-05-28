import numpy as np
import math
import types
import torch
import torch.nn as nn
import numpy as np
import cv2
import re
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from PIL import Image

def extract_first_sentence(text):
    end_index = text.find('.')
    if end_index != -1:
        first_sentence = text[:end_index + 1]
        return first_sentence.strip()
    else:
        return text.strip()
    
import re
def remove_duplicate_keywords(text, keywords): ### This function can continue to be optimized
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
    keywords = ["face", "ears", "eyes", "nose", "mouth"]
    text = remove_duplicate_keywords(text, keywords)
    key_parsing_mask_markers = ["Face", "Left_Ear", "Right_Ear", "Left_Eye", "Right_Eye", "Nose", "Upper_Lip", "Lower_Lip"]
    mapping = {
        "Face": "face",
        "Left_Ear": "ears",
        "Right_Ear": "ears",
        "Left_Eye": "eyes",
        "Right_Eye": "eyes",
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

    text_marked = text
    align_parsing_mask_list = parsing_mask_list
    for feature, marker in zip(facial_features_align[::-1], markers_align[::-1]):
        pattern = rf'\b{feature}\b'  
        text_marked_new = re.sub(pattern, f'{feature} {marker}', text_marked, count=1)
        if text_marked == text_marked_new:
            for key, value in mapping.items():
                if value == feature:
                    if key in align_parsing_mask_list:
                        del align_parsing_mask_list[key]   

        text_marked = text_marked_new 

    text_marked = text_marked.replace('\n', '')

    ordered_text = []
    text_none_makers = []
    facial_marked_count = 0
    skip_count = 0
    for marker in markers_align:
        start_idx = text_marked.find(marker)
        end_idx = start_idx + len(marker)

        while start_idx > 0 and text_marked[start_idx - 1] not in [",", ".", ";"]:
            start_idx -= 1

        while end_idx < len(text_marked) and text_marked[end_idx] not in [",", ".", ";"]:
            end_idx += 1

        context = text_marked[start_idx:end_idx].strip()
        if context == "":
            text_none_makers.append(text_marked[:end_idx])
        else:
            if skip_count!=0:
                skip_count -= 1 
                continue
            else:
                ordered_text.append(context + ",") 
                text_delete_makers = text_marked[:start_idx] + text_marked[end_idx:]
                text_marked = text_delete_makers
                facial_marked_count += 1

    align_marked_text = " ".join(ordered_text)
    replace_list = ["<|face|>", "<|ears|>", "<|nose|>", "<|eyes|>", "<|mouth|>"] 
    for item in replace_list:
        align_marked_text = align_marked_text.replace(item, "<|facial|>")

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

def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    object_segmaps = F.interpolate(object_segmaps, size=(size, size), mode="bilinear", antialias=True)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )

    num_heads = bxh // b
    cross_attention_scores = cross_attention_scores.view(b, num_heads, num_noise_latents, num_text_tokens)

    
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects
        ),
    )
    object_segmaps = (
        object_segmaps.permute(0, 2, 1)
        .unsqueeze(1)
        .expand(b, num_heads, num_noise_latents, max_num_objects)
    )
    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):  
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers

def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import Attention

    UNET_LAYER_NAMES = [ 
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers   
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores 

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn1" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
 
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )
    return unet
    
class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return background_loss - object_loss

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

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """

        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class FacePerceiverResampler(torch.nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.proj_in = torch.nn.Linear(embedding_dim, dim)
        self.proj_out = torch.nn.Linear(dim, output_dim)
        self.norm_out = torch.nn.LayerNorm(output_dim)
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)
  
class ProjPlusModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )
        
    def forward(self, id_embeds, clip_embeds, shortcut=False, scale=1.0):

        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x) 
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out
    
class AttentionMLP(nn.Module):
    def __init__(
        self,
        dtype=torch.float16,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        single_num_tokens=1,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
        max_seq_len: int = 257*2,
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.single_num_tokens = single_num_tokens
        self.latents = nn.Parameter(torch.randn(1, self.single_num_tokens, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


