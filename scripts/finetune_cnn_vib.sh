#!/usr/bin/env bash
# Fine-tune a pruned CNN-VIB-LSTM model.
# Usage: bash scripts/finetune_cnn_vib.sh <dataset_path> <finetune_model> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
FINETUNE_MODEL="${2:?Error: path to pruned model required as \$2}"
SAVE_DIR="${3:-output/cnn_vib_finetuned}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 32 --batch_size 32 \
  --img_dim1 299 --img_dim2 299 --featureExtractor Inc \
  --hidden_dim 2048 --finetune_model "$FINETUNE_MODEL" \
  --lr 1e-5 --ib_lr 5e-2 --kml 4 \
  --save_dir "$SAVE_DIR"
