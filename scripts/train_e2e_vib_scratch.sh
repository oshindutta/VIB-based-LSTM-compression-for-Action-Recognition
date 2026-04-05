#!/usr/bin/env bash
# Train and prune an end-to-end LSTM model from scratch using VIB.
# Usage: bash scripts/train_e2e_vib_scratch.sh <dataset_path> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
SAVE_DIR="${2:-output/e2e_vib_scratch}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 8 --batch_size 128 \
  --img_dim1 160 --img_dim2 120 --e2e True \
  --hidden_dim 256 --lr 1e-5 --ib_lr 5e-2 --kml 2 \
  --save_dir "$SAVE_DIR"
