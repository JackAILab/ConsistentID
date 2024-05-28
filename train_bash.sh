accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11130 \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --multi_gpu \
    train.py \
    --save_steps 1000 \
    --train_batch_size 2 \
    --data_json_file "./JSON_all.json" \
    --data_root_path "./FGID_resize" \
    --faceid_root_path "./FGID_faceID"  \
    --parsing_root_path "./FGID_parsing_mask" 
