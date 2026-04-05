#!/usr/bin/env bash
# Prune a pre-trained end-to-end LSTM model using VIB.
# Usage: bash scripts/train_e2e_vib_pretrained.sh <dataset_path> <checkpoint_model> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
CHECKPOINT_MODEL="${2:?Error: path to pre-trained model required as \$2}"
SAVE_DIR="${3:-output/e2e_vib_pretrained}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 8 --batch_size 128 \
  --img_dim1 160 --img_dim2 120 --e2e True \
  --hidden_dim 256 --checkpoint_model "$CHECKPOINT_MODEL" \
  --lr 1e-5 --ib_lr 5e-2 --kml 2 \
  --save_dir "$SAVE_DIR"
