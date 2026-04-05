#!/usr/bin/env bash
# Train and prune a CNN-LSTM model from scratch using VIB.
# Usage: bash scripts/train_cnn_vib_scratch.sh <dataset_path> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
SAVE_DIR="${2:-output/cnn_vib_scratch}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 32 --batch_size 32 \
  --img_dim1 299 --img_dim2 299 --featureExtractor Inc \
  --hidden_dim 2048 --lr 1e-5 --ib_lr 5e-2 --kml 4 \
  --save_dir "$SAVE_DIR"
