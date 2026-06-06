#!/usr/bin/env bash
# デモ用: 5つのテストサンプルを順番に実行
# 使い方: bash demo_samples.sh [sample番号 1-5]

LORA="checkpoints/lora_convert_v4/best"
BASE_WAV="/tmp/good_test_wavs"

run_sample() {
    local idx=$1
    local wav=$2
    local text=$3
    echo ""
    echo "===== サンプル ${idx} ====="
    uv run python demo_realtime.py \
        --lora_path "$LORA" \
        --wav "$wav" \
        --dialect_text "$text"
    echo ""
    read -p "[Enter]で次のサンプルへ..." _
}

# 番号指定で1件だけ実行
if [ "$1" = "1" ]; then
    run_sample 1 "${BASE_WAV}/idx13.wav" "ホットケーキんためにベーキングパウダーもちゃんと買うた."
elif [ "$1" = "2" ]; then
    run_sample 2 "${BASE_WAV}/idx22.wav" "民間と組むっちゅう手法はやり方次第。"
elif [ "$1" = "3" ]; then
    run_sample 3 "${BASE_WAV}/idx28.wav" "食費はかさむ一方たい."
elif [ "$1" = "4" ]; then
    run_sample 4 "${BASE_WAV}/idx30.wav" "ボクが買ったんは大学入ってからやねんな."
elif [ "$1" = "5" ]; then
    run_sample 5 "${BASE_WAV}/idx46.wav" "気持ち切り替えて,落ち着いて仕事しよ."
else
    # 全件順番に実行
    run_sample 1 "${BASE_WAV}/idx13.wav" "ホットケーキんためにベーキングパウダーもちゃんと買うた."
    run_sample 2 "${BASE_WAV}/idx22.wav" "民間と組むっちゅう手法はやり方次第。"
    run_sample 3 "${BASE_WAV}/idx28.wav" "食費はかさむ一方たい."
    run_sample 4 "${BASE_WAV}/idx30.wav" "ボクが買ったんは大学入ってからやねんな."
    run_sample 5 "${BASE_WAV}/idx46.wav" "気持ち切り替えて,落ち着いて仕事しよ."
fi
