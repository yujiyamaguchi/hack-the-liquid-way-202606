"""
独立した第三者ASR(kotoba-whisper-v2.1, 日本語特化Whisper)を「審査員」に使い、
(a) クリーンな参照音声(edge-tts合成、学習ラベルと同じ)
(b) LoRA生成音声(audio2audio_full)
の両方を同条件で文字起こしし、CERを比較する。

LFM2.5-Audioベースモデルのtranscribe経路がedge-tts系音声を正しく扱えない
(クリーン参照でもCER 87%)ことが判明したため、独立ジャッジに差し替えて
「生成音声の中身は実際どの程度の品質か」を公平に定量評価する。

Usage:
    uv run python check_audio_with_whisper.py
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

WHISPER_MODEL = "kotoba-tech/kotoba-whisper-v2.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATED_RESULTS = Path("eval_audio2audio_full_results/results_audio2audio_full.jsonl")
REF_AUDIO_DIR = Path("sanity_check_tts_audio")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def cer(hyp: str, ref: str) -> float:
    h = list(hyp.replace(" ", "").replace("、", "").replace("。", "").replace(",", "").replace(".", ""))
    r = list(ref.replace(" ", "").replace("、", "").replace("。", "").replace(",", "").replace(".", ""))
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(r) + 1))
    for i, ch in enumerate(h):
        ndp = [i + 1] + [0] * len(r)
        for j, cr in enumerate(r):
            ndp[j + 1] = min(dp[j] + (0 if ch == cr else 1), dp[j + 1] + 1, ndp[j] + 1)
        dp = ndp
    return dp[-1] / len(r)


def main():
    print(f"Loading independent ASR judge: {WHISPER_MODEL}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=WHISPER_MODEL,
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )

    def transcribe(wav_path: str) -> str:
        wav, sr = sf.read(wav_path, dtype="float32")
        result = asr({"array": wav, "sampling_rate": sr}, generate_kwargs={"language": "japanese", "task": "transcribe"})
        return normalize_text(result["text"])

    rows = [json.loads(l) for l in open(GENERATED_RESULTS, encoding="utf-8")]

    # --- (a) クリーン参照音声 (edge-tts) を判定 ---
    print("\n=== (a) クリーン参照音声(edge-tts) を kotoba-whisper で文字起こし ===")
    ref_scores = []
    for i, row in enumerate(rows[:6]):
        ref_wav = REF_AUDIO_DIR / f"ref_tts_{i+1}.wav"
        if not ref_wav.exists():
            continue
        ref = normalize_text(row["ref_text"])
        hyp = transcribe(str(ref_wav))
        score = cer(hyp, ref)
        ref_scores.append(score)
        print(f"[{i+1}] CER={score:.3f}")
        print(f"  Whisper書起: {hyp[:60]}")
        print(f"  参照:        {ref[:60]}")

    mean_ref = float(np.mean(ref_scores)) if ref_scores else 0.0
    print(f"\n--- クリーン参照音声(edge-tts)の Whisper CER 平均: {mean_ref:.4f} ({mean_ref*100:.1f}%) ---")

    # --- (b) LoRA生成音声 を判定 ---
    print("\n=== (b) LoRA生成音声(audio2audio_full) を kotoba-whisper で文字起こし ===")
    gen_scores = []
    for row in rows:
        wav_path = row.get("wav_path")
        if not wav_path or not Path(wav_path).exists():
            continue
        ref = normalize_text(row["ref_text"])
        hyp = transcribe(wav_path)
        score = cer(hyp, ref)
        gen_scores.append(score)
        print(f"[{row['sample']}] CER={score:.3f}")
        print(f"  Whisper書起: {hyp[:60]}")
        print(f"  参照:        {ref[:60]}")

    mean_gen = float(np.mean(gen_scores)) if gen_scores else 0.0
    print(f"\n--- LoRA生成音声の Whisper CER 平均: {mean_gen:.4f} ({mean_gen*100:.1f}%) ---")

    print(f"\n=== まとめ ===")
    print(f"クリーン参照音声(edge-tts)  Whisper-CER: {mean_ref*100:.1f}%  ({len(ref_scores)}件)  <- 判定者の健全性チェック")
    print(f"LoRA生成音声(audio2audio)   Whisper-CER: {mean_gen*100:.1f}%  ({len(gen_scores)}件)  <- 本命の品質評価")
    print("(参照側が低く、生成側が高ければ「生成音声の品質に真の問題がある」と判定できる)")
    print("(両者が近ければ「生成音声はクリーン参照と遜色ない」と判定できる)")


if __name__ == "__main__":
    main()
