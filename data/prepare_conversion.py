"""
Prepare dialect-to-standard-Japanese conversion dataset for liquid-audio fine-tuning.

Uses Qwen3-generated pairs from dialect_standard_pairs.jsonl combined with
the original dialect audio to create audio->standard-text training data.

Usage:
    uv run python data/prepare_conversion.py --split train --output data/dialect_convert_train
    uv run python data/prepare_conversion.py --split val   --output data/dialect_convert_val
    uv run python data/prepare_conversion.py --split test  --output data/dialect_convert_test
"""
from __future__ import annotations

import argparse
import io
import json
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import soundfile
import torch
from datasets import Audio, concatenate_datasets, load_dataset

from liquid_audio import LFM2AudioProcessor
from liquid_audio.data.mapper import LFM2AudioChatMapper
from liquid_audio.data.preprocess import preprocess_dataset
from liquid_audio.data.types import AudioSegment, ChatMessage, TextSegment

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
CONTEXT_LENGTH = 512

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


def load_pairs(pairs_path: Path) -> dict[str, str]:
    """Load dialect->standard mapping from JSONL."""
    pairs = {}
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs[normalize_text(obj["dialect"])] = obj["standard"]
    print(f"Loaded {len(pairs)} dialect->standard pairs from {pairs_path}")
    return pairs


class DialectConversionSamples:
    """Yields ChatMessage lists for audio->standard-Japanese conversion training."""

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

        if self.max_samples:
            ds_split = ds_split.select(range(min(self.max_samples, len(ds_split))))

        skipped = 0
        for sample in ds_split:
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
                print(f"  Warning: {e}")
                continue

            yield [
                ChatMessage(role="system", content=[TextSegment(text=SYSTEM_PROMPT)]),
                ChatMessage(role="user",   content=[AudioSegment(audio=audio_bytes)]),
                ChatMessage(role="assistant", content=[TextSegment(text=standard_text + "<|text_end|>")]),
            ]

        print(f"  Skipped {skipped} samples without standard pair")


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

    data = DialectConversionSamples(
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
    parser.add_argument("--output",  default="data/dialect_convert_train")
    parser.add_argument("--split",   default="train", choices=["train", "val", "test"])
    parser.add_argument("--pairs",   default="data/dialect_standard_pairs.jsonl")
    parser.add_argument("--dialects",default="osaka,kumamoto")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=CONTEXT_LENGTH)
    args = parser.parse_args()
    main(args)
