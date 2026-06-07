"""
round-trip検証メソッド自体の健全性チェック:
生成音声ではなく「edge-ttsで合成したクリーンな参照音声(学習ラベルと同じ合成方法)」を
ベースモデル(LoRAなし)で文字起こしし、CERを測る。

これが低ければ「round-trip transcribeパイプライン自体は機能しており、
生成音声側に真の品質問題がある」と判定できる。
高ければ「測定方法(ベースモデルのASR / Mimi系音声に対する書き起こし能力)に
問題がある可能性がある」と判定する。

Usage:
    uv run python sanity_check_roundtrip_method.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
TRANSCRIBE_PROMPT = "次の音声を文字に書き起こしてください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TTS_VOICE = "ja-JP-NanamiNeural"
TTS_SR = 24000
RESULTS_FILE = Path("eval_audio2audio_full_results/results_audio2audio_full.jsonl")


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


async def _tts_save(text: str, path: str) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice=TTS_VOICE).save(path)


def synthesize_to_wav(text: str, wav_path: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name
    asyncio.run(_tts_save(text, mp3_path))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path, "-ac", "1", "-ar", str(TTS_SR), wav_path],
        check=True,
    )
    Path(mp3_path).unlink(missing_ok=True)


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
    rows = [json.loads(l) for l in open(RESULTS_FILE, encoding="utf-8")][:6]

    print(f"Loading base model (no LoRA): {MODEL_ID}")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    model = LFM2AudioModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16).eval()

    out_dir = Path("sanity_check_tts_audio")
    out_dir.mkdir(exist_ok=True)

    scores = []
    for i, row in enumerate(rows):
        ref = normalize_text(row["ref_text"])
        wav_path = str(out_dir / f"ref_tts_{i+1}.wav")
        print(f"\n[{i+1}] synthesizing reference audio via edge-tts: {ref[:40]}")
        synthesize_to_wav(ref, wav_path)

        wav, sr = sf.read(wav_path, dtype="float32")
        hyp = transcribe(model, processor, wav, sr)
        score = cer(hyp, ref)
        scores.append(score)
        print(f"  round-trip CER (clean edge-tts ref) = {score:.3f}")
        print(f"  文字起こし: {hyp[:60]}")
        print(f"  参照:       {ref[:60]}")

    print(f"\n=== クリーン参照音声(edge-tts)の round-trip 平均CER: {np.mean(scores):.4f} ({np.mean(scores)*100:.1f}%) — {len(scores)}件 ===")
    print("(これが低ければ round-trip 検証メソッド自体は健全 = 生成音声側に真の問題がある)")
    print("(これも高ければ ベースモデルのASRパイプライン自体に問題がある可能性)")


if __name__ == "__main__":
    main()
