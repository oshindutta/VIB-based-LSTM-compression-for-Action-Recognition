#!/usr/bin/env bash
# Resume CNN-LSTM VIB pruning from a checkpoint.
# Usage: bash scripts/resume_cnn_vib.sh <dataset_path> <checkpoint> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
CHECKPOINT="${2:?Error: path to checkpoint required as \$2}"
SAVE_DIR="${3:-output/cnn_vib_resumed}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 32 --batch_size 32 \
  --img_dim1 299 --img_dim2 299 --featureExtractor Inc \
  --hidden_dim 2048 --resume "$CHECKPOINT" \
  --lr 1e-5 --ib_lr 5e-2 --kml 4 \
  --save_dir "$SAVE_DIR"
