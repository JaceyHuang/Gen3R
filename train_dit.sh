get_free_port() {
  python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=ERROR

export WAN_PATH="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
export VGGT_PATH="facebook/VGGT-1B"
export GEO_ADAPTER_PATH="/path/to/pretrained/geometry_adapter.safetensors"
export DATASET_PATH="/path/to/dataset"
export OUTPUT_DIR="/path/to/output_dir"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_port $(get_free_port) \
  --main_process_ip localhost \
  --deepspeed_multinode_launcher standard \
  --use_deepspeed \
  ./train_dit.py \
  --pretrained_wan_name_or_path $WAN_PATH \
  --vggt_path $VGGT_PATH \
  --geo_adapter_path $GEO_ADAPTER_PATH \
  --config_path ./gen3r/config/gen3r.yaml \
  --train_data_dir $DATASET_PATH \
  --video_sample_size 560 \
  --video_sample_stride 2 \
  --video_sample_n_frames 49 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --dataloader_num_workers 8 \
  --num_train_epochs 7 \
  --checkpointing_and_validation_steps 50 \
  --learning_rate 1e-05 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 50 \
  --seed 42 \
  --output_dir $OUTPUT_DIR \
  --mixed_precision "bf16" \
  --adam_weight_decay 3e-2 \
  --adam_epsilon 1e-10 \
  --max_grad_norm 0.25 \
  --initial_grad_norm_ratio 2 \
  --uniform_sampling \
  --trainable_modules "." \
  --report_to 'tensorboard' \
  --gradient_checkpointing \
  --sanity_check \
  --report_model_info