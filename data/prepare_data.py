"""
Prepare Common Voice Japanese (elderly-filtered) as liquid-audio training data.

Follows the official hackathon recipe (scripts/audio/train.py pattern):
  Common Voice Japanese  →  LFM2AudioChatMapper  →  preprocess_dataset  →  HF dataset on disk

Usage:
    # Train split, elderly only:
    uv run python data/prepare_data.py

    # Test split (for evaluation):
    uv run python data/prepare_data.py --split test --output data/cv_elderly_test --max_samples 200

    # All ages (for baseline comparison dataset):
    uv run python data/prepare_data.py --all_ages --output data/cv_all_train
"""
from __future__ import annotations

import argparse
import io
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import soundfile
import torch
from datasets import load_dataset

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR_SYSTEM_PROMPT = "Perform ASR in japanese."
ELDERLY_AGES = {"sixties", "seventies", "eighties", "nineties"}
CONTEXT_LENGTH = 512  # tokens; shorter than TTS since no audio_out


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def audio_array_to_bytes(array, samplerate: int) -> bytes:
    buf = io.BytesIO()
    soundfile.write(buf, array, samplerate, format="WAV")
    return buf.getvalue()


class CommonVoiceASRSamples:
    """Yields list[ChatMessage] per sample. Follows the TrainingSamples pattern
    from the official hackathon audio train.py."""

    def __init__(
        self,
        split: str = "train",
        elderly_only: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.split = split
        self.elderly_only = elderly_only
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        print(f"Loading Common Voice 17 Japanese ({self.split})...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "ja",
            split=self.split,
        )

        if self.elderly_only:
            original_len = len(ds)
            ds = ds.filter(lambda x: x.get("age") in ELDERLY_AGES, num_proc=4)
            print(f"Elderly filter: {original_len} → {len(ds)} samples")

        if self.max_samples and len(ds) > self.max_samples:
            ds = ds.shuffle(seed=42).select(range(self.max_samples))
            print(f"Subsampled to {len(ds)} samples")

        for sample in ds:
            transcription = normalize_text(sample["sentence"])
            if not transcription:
                continue

            audio_info = sample["audio"]
            audio_bytes = audio_array_to_bytes(audio_info["array"], audio_info["sampling_rate"])

            yield [
                ChatMessage(role="system", content=[TextSegment(text=ASR_SYSTEM_PROMPT)]),
                ChatMessage(role="user", content=[AudioSegment(audio=audio_bytes)]),
                ChatMessage(role="assistant", content=[TextSegment(text=transcription)]),
            ]


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    if out_path.exists():
        print(f"Output path {out_path} already exists. Delete it to regenerate.")
        return

    print(f"Loading liquid-audio processor from {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device=device).eval()
    mapper = LFM2AudioChatMapper(
        processor,
        codebooks=8,
        interleaved_text_tokens=6,
        interleaved_audio_tokens=9,  # JP model ratio
    )

    data = CommonVoiceASRSamples(
        split=args.split,
        elderly_only=not args.all_ages,
        max_samples=args.max_samples,
    )

    print(f"Preprocessing to {out_path}...")
    preprocess_dataset(
        data=data,
        output_path=str(out_path),
        mapper=mapper,
        max_context_length=args.context_length,
    )

    # Free VRAM before trainer might load the model
    del processor, mapper
    if device == "cuda":
        torch.cuda.empty_cache()

    # Verify
    from datasets import load_from_disk
    ds = load_from_disk(str(out_path))
    print(f"\nDone. Saved {len(ds)} samples to {out_path}")
    print(f"Features: {ds.features}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/cv_elderly_train")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--all_ages", action="store_true")
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)
    args = parser.parse_args()
    main(args)
