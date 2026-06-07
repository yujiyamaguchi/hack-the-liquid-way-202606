"""
Prepare dialect-to-standard-Japanese AUDIO conversion dataset for liquid-audio fine-tuning.

Like prepare_conversion.py (audio -> standard TEXT), but the assistant turn
also carries a synthesized standard-Japanese AUDIO label (via edge-tts), packed
as an InterleavedSegment so the model learns to produce interleaved text+audio
output matching generate_interleaved()'s inference-time format.

Usage:
    uv run python data/prepare_conversion_audio2audio.py --split train --dialects osaka --max_samples 48 --output data/audio2audio_train
    uv run python data/prepare_conversion_audio2audio.py --split val   --dialects osaka --max_samples 6  --output data/audio2audio_val
    uv run python data/prepare_conversion_audio2audio.py --split test  --dialects osaka --max_samples 6  --output data/audio2audio_test
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import subprocess
import tempfile
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import soundfile
import torch
from datasets import Audio, concatenate_datasets, load_dataset

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, InterleavedSegment, TextSegment

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
CONTEXT_LENGTH = 512
TTS_VOICE = "ja-JP-NanamiNeural"
TTS_SAMPLE_RATE = 24000

DIALECT_DATASETS = {
    "osaka":    "federerjiang/dialect.osaka",
    "kumamoto": "federerjiang/dialect.kumamoto",
}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def audio_array_to_bytes(array, samplerate: int) -> bytes:
    buf = io.BytesIO()
    soundfile.write(buf, array, samplerate, format="WAV")
    return buf.getvalue()


async def _tts_save(text: str, path: str, voice: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(path)


def synthesize_standard_audio(text: str, voice: str = TTS_VOICE) -> bytes:
    """edge-tts で標準語テキストを音声合成 → ffmpeg で WAV 24kHz mono に変換 → bytes化"""
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(mp3_fd)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        asyncio.run(_tts_save(text, mp3_path, voice))
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", str(TTS_SAMPLE_RATE), wav_path],
            capture_output=True, check=True,
        )
        array, sr = soundfile.read(wav_path, dtype="float32")
        return audio_array_to_bytes(array, sr)
    finally:
        for p in (mp3_path, wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def load_pairs(pairs_path: Path) -> dict[str, str]:
    """Load dialect->standard mapping from JSONL."""
    pairs = {}
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs[normalize_text(obj["dialect"])] = obj["standard"]
    print(f"Loaded {len(pairs)} dialect->standard pairs from {pairs_path}")
    return pairs


class DialectAudio2AudioSamples:
    """Yields ChatMessage lists for audio->standard-Japanese-AUDIO conversion training."""

    def __init__(
        self,
        dialects: list[str],
        pairs: dict[str, str],
        split: str = "train",
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_samples: int | None = None,
    ) -> None:
        self.dialects = dialects
        self.pairs = pairs
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        datasets_list = []
        for dialect in self.dialects:
            hf_id = DIALECT_DATASETS[dialect]
            print(f"Loading {hf_id}...")
            ds = load_dataset(hf_id, split="train")
            ds = ds.cast_column("audio", Audio(decode=False))
            datasets_list.append(ds)

        combined = concatenate_datasets(datasets_list).shuffle(seed=42)
        n = len(combined)
        test_n = max(1, int(n * self.test_ratio))
        val_n  = max(1, int(n * self.val_ratio))
        if self.split == "test":
            ds_split = combined.select(range(test_n))
        elif self.split == "val":
            ds_split = combined.select(range(test_n, test_n + val_n))
        else:
            ds_split = combined.select(range(test_n + val_n, n))

        skipped = 0
        synthesized = 0
        for sample in ds_split:
            if self.max_samples and synthesized >= self.max_samples:
                break

            dialect_text = normalize_text(sample["sentence"])
            standard_text = self.pairs.get(dialect_text, "")
            if not standard_text:
                skipped += 1
                continue

            audio_info = sample["audio"]
            try:
                audio_bytes = audio_info.get("bytes") or audio_array_to_bytes(
                    audio_info["array"], audio_info["sampling_rate"]
                )
                if not audio_bytes:
                    continue
            except Exception as e:
                print(f"  Warning (input audio): {e}")
                continue

            try:
                standard_audio_bytes = synthesize_standard_audio(standard_text)
            except Exception as e:
                print(f"  Warning (TTS synth): {e}")
                continue

            synthesized += 1
            print(f"  [{synthesized}] {standard_text[:40]}")

            yield [
                ChatMessage(role="system", content=[TextSegment(text=SYSTEM_PROMPT)]),
                ChatMessage(role="user",   content=[AudioSegment(audio=audio_bytes)]),
                ChatMessage(role="assistant", content=[
                    InterleavedSegment(text=standard_text + "<|text_end|>", audio=standard_audio_bytes)
                ]),
            ]

        print(f"  Synthesized {synthesized} samples, skipped {skipped} without standard pair")


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    if out_path.exists():
        print(f"{out_path} already exists. Delete to regenerate.")
        return

    pairs = load_pairs(Path(args.pairs))
    dialects = args.dialects.split(",")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device=device).eval()
    mapper = LFM2AudioChatMapper(
        processor, codebooks=8, interleaved_text_tokens=6, interleaved_audio_tokens=9,
    )

    data = DialectAudio2AudioSamples(
        dialects=dialects,
        pairs=pairs,
        split=args.split,
        max_samples=args.max_samples,
    )

    print(f"Preprocessing to {out_path}...")
    preprocess_dataset(
        data=data,
        output_path=str(out_path),
        mapper=mapper,
        max_context_length=args.context_length,
    )

    del processor, mapper
    if device == "cuda":
        torch.cuda.empty_cache()

    from datasets import load_from_disk
    ds = load_from_disk(str(out_path))
    print(f"Done. Saved {len(ds)} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="data/audio2audio_train")
    parser.add_argument("--split",   default="train", choices=["train", "val", "test"])
    parser.add_argument("--pairs",   default="data/dialect_standard_pairs.jsonl")
    parser.add_argument("--dialects",default="osaka")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)
    args = parser.parse_args()
    main(args)
