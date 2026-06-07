"""
生成された標準語音声WAVを、ベースモデル(LoRAなし)のASRで文字起こしし、
参照テキストとのCERで「実際に正しい内容を発話できているか」を round-trip 検証する。

(再生環境がないため、聴感評価の代替として用いる客観指標)

Usage:
    uv run python check_audio2audio_roundtrip.py
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
TRANSCRIBE_PROMPT = "次の音声を文字に書き起こしてください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("eval_audio2audio_results")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def cer(hyp: str, ref: str) -> float:
    h = list(hyp.replace(" ", ""))
    r = list(ref.replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(r) + 1))
    for i, ch in enumerate(h):
        ndp = [i + 1] + [0] * len(r)
        for j, cr in enumerate(r):
            ndp[j + 1] = min(dp[j] + (0 if ch == cr else 1), dp[j + 1] + 1, ndp[j] + 1)
        dp = ndp
    return dp[-1] / len(r)


def transcribe(model, processor, wav: np.ndarray, sr: int) -> str:
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(TRANSCRIBE_PROMPT)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(torch.from_numpy(wav[None]).float().to(DEVICE), sr)
    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens: list[int] = []
    with torch.no_grad():
        for token in model.generate_interleaved(**chat, max_new_tokens=120, text_top_k=1):
            if token.numel() == 1:
                tid = token.item()
                if tid == 7:
                    break
                if tid == 130:
                    continue
                text_tokens.append(tid)
                if len(text_tokens) >= 100:
                    break
            else:
                if token[0].item() == 2048:
                    break
    if not text_tokens:
        return ""
    result = processor.text.decode(text_tokens, skip_special_tokens=True)
    return result if isinstance(result, str) else result[0]


def main():
    print(f"Loading base model (no LoRA): {MODEL_ID}")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    model = LFM2AudioModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16).eval()

    rows = [json.loads(l) for l in open(RESULTS_DIR / "results_audio2audio.jsonl", encoding="utf-8")]

    scores = []
    for row in rows:
        wav_path = row["wav_path"]
        if not wav_path or not Path(wav_path).exists():
            continue
        wav, sr = sf.read(wav_path, dtype="float32")
        hyp = transcribe(model, processor, wav, sr)
        ref = normalize_text(row["ref_text"])
        score = cer(hyp, ref)
        scores.append(score)
        print(f"[{row['sample']}] round-trip CER={score:.3f}")
        print(f"  生成音声を文字起こし: {hyp[:60]}")
        print(f"  参照(標準語テキスト): {ref[:60]}")

    if scores:
        print(f"\n=== round-trip 平均CER: {np.mean(scores):.4f} ({np.mean(scores)*100:.1f}%) — {len(scores)}件 ===")
        print("(低いほど「生成音声が意図した標準語を正しく発話できている」ことを示す)")


if __name__ == "__main__":
    main()
