#!/usr/bin/env bash
# Evaluate a trained VIB model.
# Usage: bash scripts/evaluate.sh <dataset_path> <checkpoint> <save_dir>
# Adjust --img_dim1 / --img_dim2 to match the model's expected input size.

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
CHECKPOINT="${2:?Error: path to checkpoint required as \$2}"
SAVE_DIR="${3:-output/eval}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --img_dim1 299 --img_dim2 299 --featureExtractor Inc \
  --resume "$CHECKPOINT" \
  --save_dir "$SAVE_DIR" --eval True
