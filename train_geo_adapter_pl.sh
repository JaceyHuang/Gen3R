get_free_port() {
  python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=ERROR

export VGGT_PATH="facebook/VGGT-1B"
export WAN_VAE_PATH="/path/to/pretrained/Wan2.1_VAE.pth"
export DATASET_PATH="/path/to/dataset/train_videos_dirs.txt"
export OUTPUT_DIR="/path/to/output_dir"

torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port $(get_free_port) \
  ./train_geo_adapter_pl.py \
  --use_deepspeed \
  --vggt_path $VGGT_PATH \
  --wan_vae_path $WAN_VAE_PATH \
  --dataset_path $DATASET_PATH \
  --video_sample_size 560 \
  --video_sample_stride 2 \
  --video_sample_n_frames 49 \
  --train_batch_size 1 \
  --accumulate_grad_batches 8 \
  --dataloader_num_workers 8 \
  --num_train_epochs 10 \
  --learning_rate 1e-05 \
  --seed 42 \
  --output_dir $OUTPUT_DIR \
  --report_to "wandb"