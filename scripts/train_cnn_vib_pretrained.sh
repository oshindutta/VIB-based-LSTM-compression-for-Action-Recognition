#!/usr/bin/env bash
# Prune a pre-trained CNN-LSTM model using VIB.
# Usage: bash scripts/train_cnn_vib_pretrained.sh <dataset_path> <checkpoint_model> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
CHECKPOINT_MODEL="${2:?Error: path to pre-trained model required as \$2}"
SAVE_DIR="${3:-output/cnn_vib_pretrained}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 32 --batch_size 32 \
  --img_dim1 299 --img_dim2 299 --featureExtractor Inc \
  --hidden_dim 2048 --checkpoint_model "$CHECKPOINT_MODEL" \
  --lr 1e-5 --ib_lr 5e-2 --kml 4 \
  --save_dir "$SAVE_DIR"
