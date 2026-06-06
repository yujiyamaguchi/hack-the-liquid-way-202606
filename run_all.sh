#!/bin/bash
set -eo pipefail
echo "===== run_all.sh 開始 ====="

# ── 案①テキストデータ（既に生成済みの場合はスキップ）──────────────
echo "[1/6] 案①テキストデータ確認..."
if [ ! -d data/dialect_text_train ]; then
    uv run python data/prepare_conversion_text.py --split train --output data/dialect_text_train > logs/prepare_text_train.log 2>&1 &
    PID_TEXT_TRAIN=$!
    uv run python data/prepare_conversion_text.py --split val   --output data/dialect_text_val   > logs/prepare_text_val.log   2>&1 &
    PID_TEXT_VAL=$!
    uv run python data/prepare_conversion_text.py --split test  --output data/dialect_text_test  > logs/prepare_text_test.log  2>&1 &
    PID_TEXT_TEST=$!
else
    echo "  案①データ既存 (スキップ)"
    PID_TEXT_TRAIN="" PID_TEXT_VAL="" PID_TEXT_TEST=""
fi

# ── 案②（音声入力）学習: lora_convert_v4 ──────────────────────
echo "[2/6] 案②学習開始 (lora_convert_v4, 1000 steps)..."
TRAIN_DATA=data/dialect_convert_train \
VAL_DATA=data/dialect_convert_val \
OUTPUT_DIR=checkpoints/lora_convert_v4 \
MAX_STEPS=1000 \
LR=1e-4 \
uv run python train.py 2>&1 | tee logs/train_convert_v4.log
echo "[2/6] 案②学習完了"

# ── 案①テキストデータ生成完了待ち ────────────────────────────
echo "[3/6] 案①データ生成待ち..."
[ -n "$PID_TEXT_TRAIN" ] && wait $PID_TEXT_TRAIN $PID_TEXT_VAL $PID_TEXT_TEST
echo "[3/6] 案①データ確認完了"

# ── 案①（テキスト入力）学習: lora_text_v1 ─────────────────────
echo "[4/6] 案①学習開始 (lora_text_v1, 1000 steps)..."
TRAIN_DATA=data/dialect_text_train \
VAL_DATA=data/dialect_text_val \
OUTPUT_DIR=checkpoints/lora_text_v1 \
MAX_STEPS=1000 \
LR=1e-4 \
uv run python train.py 2>&1 | tee logs/train_text_v1.log
echo "[4/6] 案①学習完了"

# ── 評価: 案② ─────────────────────────────────────────────────
echo "[5/6] 案②評価 (音声入力, n=50)..."
uv run python eval_convert.py \
  --lora_path checkpoints/lora_convert_v4/best \
  --input_mode audio \
  --n 50 \
  --output logs/eval_convert_v4.jsonl \
  2>&1 | tee logs/eval_convert_v4.log
echo "[5/6] 案②評価完了"

# ── 評価: 案① ─────────────────────────────────────────────────
echo "[6/6] 案①評価 (テキスト入力, n=50)..."
uv run python eval_convert.py \
  --lora_path checkpoints/lora_text_v1/best \
  --input_mode text \
  --n 50 \
  --output logs/eval_text_v1.jsonl \
  2>&1 | tee logs/eval_text_v1.log
echo "[6/6] 案①評価完了"

echo ""
echo "===== 全完了 ====="
echo "案② CER:"
grep "平均CER" logs/eval_convert_v4.log || true
echo "案① CER:"
grep "平均CER" logs/eval_text_v1.log || true
