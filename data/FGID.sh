export CUDA_VISIBLE_DEVICES=0

python ./FGID_mask.py

python ./FGID_Caption.py

python ./FGID_faceid_embeds.py

python ./FGID_fuse_JSON.py

