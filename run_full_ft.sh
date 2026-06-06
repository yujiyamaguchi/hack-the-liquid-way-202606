#!/bin/bash
set -eo pipefail
echo "v5完了待ち中..."
until grep -q "^Done\." logs/train_convert_v5.log 2>/dev/null; do sleep 30; done
echo "v5完了！Full FT開始 (800 steps, LR=3e-5)..."
NO_LORA=1 \
TRAIN_DATA=data/dialect_convert_train \
VAL_DATA=data/dialect_convert_val \
OUTPUT_DIR=checkpoints/full_ft_v1 \
MAX_STEPS=800 \
LR=3e-5 \
uv run python train.py 2>&1 | tee logs/train_full_ft_v1.log
echo "Full FT完了！"
