#!/usr/bin/env bash
# Convert a VIB sparse model to a dense-compressed model.
# Usage: bash scripts/convert_to_dense.sh <vib_model> <dataset_path>

VIB_MODEL="${1:?Error: path to VIB model required as \$1}"
DATASET_PATH="${2:-UCF11_updated_mpg-frames}"

python modelport.py \
  --VIBmodel "$VIB_MODEL" \
  --dataset_path "$DATASET_PATH" \
  --latent_dim 2048 --hidden_dim 2048 --featureExtractor Inc
