### Training from coarse to fine

accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11130 \
    --num_processes 2 \
    --gpu_ids 6,7 \
    --multi_gpu \
    train_SDXL.py \
    --save_steps 5000 \
    --train_batch_size 2 \
    --num_train_epochs 5 \
    --learning_rate=1e-04 \
    --weight_decay=0.01 \
    --output_dir "./outputs/faceid_plus/" \
    --pretrained_model_name_or_path "./stable-diffusion-xl-base-1.0" \
    --data_json_file "./JSON_all.json" \
    --data_root_path "./FGID_resize" \
    --faceid_root_path "./FGID_faceID"  \
    --parsing_root_path "./FGID_parsing_mask" 






