"""
方言音声 → 標準語「音声」直接生成デモ (lora_audio2audio_full_v1)

lora_convert_v4 (テキスト変換 + edge-ttsで音声合成) とは異なり、
このモデルは方言音声から標準語の「テキスト」と「音声」を同時に直接生成する
(edge-tts不使用、モデル自身の声)。画面にはモデル自身の標準語テキスト出力を
そのまま表示する。

サンプルは lora_convert_v4 の評価と同一の test split (osaka+kumamoto, seed=42)
から、find_best_demo_samples.py で広く探索して見つけた、変換結果が自然で
安定していた代表3件を選ぶ。

Usage:
    uv run python demo_audio2audio.py
    uv run python demo_audio2audio.py --n 3
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset

from demo_realtime import play_audio_powershell, prewarm_audio
from eval_audio2audio_full import load_model_with_lora, normalize_text, predict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_SR = 24000

# find_best_demo_samples.py で40件を広く探索し、方言→標準語の変換結果
# (モデル自身のテキスト出力・生成音声とも) が自然で安定していた代表3サンプル
# (test split内の出現順, 1-indexed)
SAMPLE_INDICES = [9, 31, 29]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="checkpoints/lora_audio2audio_full_v1/best")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    model, processor = load_model_with_lora(args.lora_path)

    print("テストデータロード中 (osaka+kumamoto combined, seed=42 — lora_convert_v4と同一split)...")
    ds_osaka = load_dataset("federerjiang/dialect.osaka", split="train")
    ds_kuma = load_dataset("federerjiang/dialect.kumamoto", split="train")
    combined = concatenate_datasets([ds_osaka, ds_kuma]).shuffle(seed=42)
    test_n = max(1, int(len(combined) * 0.1))
    ds_test = combined.select(range(test_n)).cast_column("audio", Audio(sampling_rate=16000))

    print("音声デバイス初期化中...")
    prewarm_audio()

    print("\n" + "=" * 58)
    print("  方言音声 → 標準語「音声」 直接生成デモ")
    print("  (LFM2.5-Audio-1.5B-JP + LoRA / edge-tts不使用・モデル自身の声)")
    print("=" * 58)

    target = SAMPLE_INDICES[: args.n]
    target_set = set(target)
    shown = 0
    i = 0
    for sample in ds_test:
        if shown >= len(target):
            break
        i += 1
        if i not in target_set:
            continue
        dialect_text = normalize_text(sample["sentence"])
        shown += 1

        audio = sample["audio"]["array"].astype(np.float32)

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ サンプル {shown}/{len(target)} ━━━")
        print(f"  【方言音声(入力)】 {dialect_text}")
        print(f"  ({len(audio) / 16000:.1f}秒)")
        print("\n[再生] 方言音声(原音)...")
        peak = np.abs(audio).max()
        audio_norm = audio / (peak + 1e-8) * 0.9 if peak > 0 else audio
        play_audio_powershell(audio_norm, 16000)

        print("[生成] LFM2.5-Audio-1.5B-JP + LoRA で標準語の「テキスト」と「音声」を直接生成中...")
        text, waveform, elapsed = predict(model, processor, audio)

        print(f"\n  【標準語(モデルのテキスト出力)】 {text}")
        print(f"  推論時間: {elapsed:.1f}秒")

        if waveform is not None:
            wav_np = waveform[0].float().cpu().numpy()
            print(f"\n[再生] 生成された標準語音声 ({wav_np.shape[-1] / GEN_SR:.1f}秒)...")
            play_audio_powershell(wav_np, GEN_SR)
        else:
            print("[警告] 音声フレームが生成されませんでした")

    print(f"\n完了 ({shown}件)。")


if __name__ == "__main__":
    main()
