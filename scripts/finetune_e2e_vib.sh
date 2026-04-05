#!/usr/bin/env bash
# Fine-tune a pruned end-to-end VIB-LSTM model.
# Usage: bash scripts/finetune_e2e_vib.sh <dataset_path> <finetune_model> <save_dir>

DATASET_PATH="${1:-UCF11_updated_mpg-frames}"
FINETUNE_MODEL="${2:?Error: path to pruned model required as \$2}"
SAVE_DIR="${3:-output/e2e_vib_finetuned}"

python train_VIB.py \
  --dataset_path "$DATASET_PATH" \
  --num_epochs 100 --sequence_length 8 --batch_size 128 \
  --img_dim1 160 --img_dim2 120 --e2e True \
  --hidden_dim 256 --finetune_model "$FINETUNE_MODEL" \
  --lr 1e-5 --ib_lr 5e-2 --kml 2 \
  --save_dir "$SAVE_DIR"
