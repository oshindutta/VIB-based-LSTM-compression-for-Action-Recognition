#!/usr/bin/env bash
# Resume end-to-end LSTM VIB pruning from a checkpoint.
# Usage: bash scripts/resume_e2e_vib.sh <dataset_path> <checkpoint> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
CHECKPOINT="${2:?Error: path to checkpoint required as \$2}"
SAVE_DIR="${3:-output/e2e_vib_resumed}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 8 --batch_size 128 \
  --img_dim1 160 --img_dim2 120 --e2e True \
  --hidden_dim 256 --resume "$CHECKPOINT" \
  --lr 1e-5 --ib_lr 5e-2 --kml 2 \
  --save_dir "$SAVE_DIR"
