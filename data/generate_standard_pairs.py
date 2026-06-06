"""
Generate standard Japanese (標準語) pairs from dialect text using Qwen3-32B local API.

Output: data/dialect_standard_pairs.jsonl
  {"dialect": "...", "standard": "...", "source": "osaka"|"kumamoto"}

Usage:
    uv run python data/generate_standard_pairs.py
    uv run python data/generate_standard_pairs.py --workers 8 --output data/dialect_standard_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from openai import OpenAI
from tqdm import tqdm

API_BASE  = "http://127.0.0.1:8006/v1"
API_KEY   = "dummy"
MODEL     = "ggml-org/Qwen3-32B-GGUF"
SYSTEM    = (
    "あなたは日本語の方言を標準語に変換する専門家です。"
    "方言テキストを自然な標準語（です・ます調）に変換してください。"
    "変換後のテキストのみを返してください。余分な説明は不要です。 /no_think"
)


def convert_one(client: OpenAI, dialect: str, source: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": dialect},
                ],
                max_tokens=256,
                temperature=0.3,
            )
            standard = resp.choices[0].message.content.strip()
            if not standard:
                continue
            return {"dialect": dialect, "standard": standard, "source": source}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"\nFailed after {retries} retries: {dialect[:40]}... ({e})")
    return None


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all samples
    ds_o = load_dataset("federerjiang/dialect.osaka",    split="train")
    ds_k = load_dataset("federerjiang/dialect.kumamoto", split="train")
    items = (
        [(row["sentence"], "osaka")    for row in ds_o] +
        [(row["sentence"], "kumamoto") for row in ds_k]
    )
    # Deduplicate
    seen = set()
    unique_items = []
    for text, src in items:
        if text and text not in seen:
            seen.add(text)
            unique_items.append((text, src))
    print(f"Total unique sentences: {len(unique_items)}")

    # Resume from existing output
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj["dialect"])
                except Exception:
                    pass
        print(f"Resuming: {len(done)} already done")

    remaining = [(t, s) for t, s in unique_items if t not in done]
    print(f"Remaining: {len(remaining)}")

    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    with open(out_path, "a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert_one, client, text, src): text
                for text, src in remaining
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                result = fut.result()
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()

    # Summary
    total = sum(1 for _ in open(out_path))
    print(f"\nDone. {total} pairs saved to {out_path}")

    # Quick quality check (5 samples)
    print("\n=== Sample check ===")
    with open(out_path) as f:
        for i, line in enumerate(f):
            if i >= 5: break
            obj = json.loads(line)
            print(f"[{obj['source']}] 方言: {obj['dialect']}")
            print(f"         標準語: {obj['standard']}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="data/dialect_standard_pairs.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    main(args)
