"""
Prepare Japanese dialect speech datasets for liquid-audio ASR fine-tuning.

Uses:
  - federerjiang/dialect.osaka  (大阪弁 / Osaka-ben, 1,300 utterances)
  - federerjiang/dialect.kumamoto (熊本弁 / Kumamoto-ben, 1,300 utterances)

Usage:
    # Prepare train set (both dialects combined):
    uv run python data/prepare_dialect.py --output data/dialect_train

    # Prepare test set for evaluation:
    uv run python data/prepare_dialect.py --split test --output data/dialect_test --max_samples 200

    # Only Osaka:
    uv run python data/prepare_dialect.py --dialects osaka --output data/dialect_osaka_train
"""
from __future__ import annotations

import argparse
import io
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import soundfile
import torch
from datasets import Audio, load_dataset, concatenate_datasets

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR_SYSTEM_PROMPT = "Perform ASR in japanese."
CONTEXT_LENGTH = 512

DIALECT_DATASETS = {
    "osaka": "federerjiang/dialect.osaka",
    "kumamoto": "federerjiang/dialect.kumamoto",
}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def audio_array_to_bytes(array, samplerate: int) -> bytes:
    buf = io.BytesIO()
    soundfile.write(buf, array, samplerate, format="WAV")
    return buf.getvalue()


class DialectASRSamples:
    """Yields list[ChatMessage] per dialect sample.

    Combines multiple dialect datasets into a single stream.
    The 'sentence' field in these datasets is the standard Japanese
    transcription of dialectal speech — ideal for ASR training.
    """

    def __init__(
        self,
        dialects: list[str],
        split: str = "train",
        max_samples: int | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        self.dialects = dialects
        self.split = split
        self.max_samples = max_samples
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def __iter__(self) -> Iterator[list[ChatMessage]]:
        datasets_list = []
        for dialect in self.dialects:
            if dialect not in DIALECT_DATASETS:
                print(f"Warning: unknown dialect '{dialect}', skipping")
                continue
            hf_id = DIALECT_DATASETS[dialect]
            print(f"Loading {hf_id}...")
            ds = load_dataset(hf_id, split="train")
            # Keep raw audio bytes — liquid-audio decodes them internally
            ds = ds.cast_column("audio", Audio(decode=False))
            print(f"  {dialect}: {len(ds)} samples")
            datasets_list.append(ds)

        if not datasets_list:
            raise ValueError(f"No valid dialect datasets found for: {self.dialects}")

        combined = concatenate_datasets(datasets_list).shuffle(seed=42)

        # 3-way split: test | val | train  (indices from the front)
        n = len(combined)
        test_n = max(1, int(n * self.test_ratio))
        val_n  = max(1, int(n * self.val_ratio))
        if self.split == "test":
            ds_split = combined.select(range(test_n))
        elif self.split == "val":
            ds_split = combined.select(range(test_n, test_n + val_n))
        else:  # train
            ds_split = combined.select(range(test_n + val_n, n))

        if self.max_samples and len(ds_split) > self.max_samples:
            ds_split = ds_split.select(range(self.max_samples))

        print(f"Using {len(ds_split)} samples ({self.split} split)")

        for sample in ds_split:
            transcription = normalize_text(sample["sentence"])
            if not transcription:
                continue

            audio_info = sample["audio"]
            try:
                # With Audio(decode=False), audio_info["bytes"] contains raw bytes
                audio_bytes = audio_info.get("bytes") or audio_array_to_bytes(
                    audio_info["array"], audio_info["sampling_rate"]
                )
                if not audio_bytes:
                    continue
            except Exception as e:
                print(f"  Warning: audio conversion failed: {e}")
                continue

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

    dialects = args.dialects.split(",")
    print(f"Preparing dialect ASR data: {dialects} ({args.split} split)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading liquid-audio processor from {MODEL_ID} (device={device})...")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID, device=device).eval()
    mapper = LFM2AudioChatMapper(
        processor,
        codebooks=8,
        interleaved_text_tokens=6,
        interleaved_audio_tokens=9,
    )

    data = DialectASRSamples(
        dialects=dialects,
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
    print(f"\nDone. Saved {len(ds)} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/dialect_train")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--dialects", default="osaka,kumamoto", help="Comma-separated: osaka,kumamoto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)
    args = parser.parse_args()
    main(args)
