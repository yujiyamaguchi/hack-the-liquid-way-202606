"""
デモ用に「成功している(=Whisper-CERが低い、理想的にはゼロ)」サンプルを
広い範囲から探索するツール。

audio2audio_full LoRA でテストsplitの先頭N件を生成し、独立ASR(kotoba-whisper-v2.1)
で生成音声を文字起こし、正解とのCERでソートして上位を報告する。
ここで見つかった良サンプルの index を demo_audio2audio.py の SAMPLE_INDICES に
反映してデモを構成する。

Usage:
    uv run python find_best_demo_samples.py --n 40
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import pipeline

from eval_audio2audio_full import (
    cer,
    load_model_with_lora,
    load_pairs,
    normalize_text,
    predict,
)

WHISPER_MODEL = "kotoba-tech/kotoba-whisper-v2.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_SR = 24000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="checkpoints/lora_audio2audio_full_v1/best")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()

    model, processor = load_model_with_lora(args.lora_path)
    pairs = load_pairs()

    print(f"独立ASR審査員ロード中: {WHISPER_MODEL}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )

    def whisper_transcribe(wav: np.ndarray, sr: int) -> str:
        result = asr({"array": wav, "sampling_rate": sr}, generate_kwargs={"language": "japanese", "task": "transcribe"})
        return normalize_text(result["text"])

    print("テストデータロード中 (osaka+kumamoto combined, seed=42)...")
    ds_osaka = load_dataset("federerjiang/dialect.osaka", split="train")
    ds_kuma = load_dataset("federerjiang/dialect.kumamoto", split="train")
    combined = concatenate_datasets([ds_osaka, ds_kuma]).shuffle(seed=42)
    test_n = max(1, int(len(combined) * 0.1))
    ds_test = combined.select(range(test_n)).cast_column("audio", Audio(sampling_rate=16000))

    print(f"探索開始 (先頭 {args.n} 件、ペア対応のあるもののみ評価)...\n")

    results = []
    i = 0
    for sample in ds_test:
        if i >= args.n:
            break
        i += 1
        dialect_text = normalize_text(sample["sentence"])
        standard_ref = pairs.get(dialect_text, "")
        if not standard_ref:
            continue

        audio = sample["audio"]["array"].astype(np.float32)
        text, waveform, elapsed = predict(model, processor, audio)

        whisper_text = ""
        whisper_cer = 1.0
        if waveform is not None:
            wav_np = waveform[0].float().cpu().numpy()
            whisper_text = whisper_transcribe(wav_np, GEN_SR)
            whisper_cer = cer(whisper_text, standard_ref)

        results.append({
            "index": i,
            "whisper_cer": whisper_cer,
            "dialect": dialect_text,
            "model_text": text,
            "whisper_text": whisper_text,
            "ref": standard_ref,
        })
        print(f"[{i}] Whisper-CER={whisper_cer:.3f}  方言: {dialect_text[:30]}")
        print(f"      Whisper書起: {whisper_text[:50]}")
        print(f"      正解:        {standard_ref[:50]}")

    results.sort(key=lambda r: r["whisper_cer"])
    print(f"\n=== Whisper-CER 昇順 上位候補 (探索 {len(results)} 件中) ===")
    for r in results[:15]:
        print(f"\n[index={r['index']}] Whisper-CER={r['whisper_cer']:.3f}")
        print(f"  方言:        {r['dialect']}")
        print(f"  モデル出力:  {r['model_text']}")
        print(f"  Whisper書起: {r['whisper_text']}")
        print(f"  正解:        {r['ref']}")

    n_perfect = sum(1 for r in results if r["whisper_cer"] == 0.0)
    print(f"\nCER=0.000(完全一致)件数: {n_perfect} / {len(results)}")


if __name__ == "__main__":
    main()
