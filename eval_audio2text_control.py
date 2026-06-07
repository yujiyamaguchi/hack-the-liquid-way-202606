"""
裏取り用コントロール評価: 方言音声→標準語「テキスト」LoRA (lora_audio2text_control_v1)

audio2audio実験と全く同じ60件(train48/val6/test6, osaka, seed=42)で学習したモデルを、
eval_audio2audio.py と同一のテスト分割・同一6サンプルで評価する。
"収束しない・精度が出ないのはデータが特殊だから" という可能性を排除し、
"音声出力が難しいから" という結論の裏付けとする。

Usage:
    uv run python eval_audio2text_control.py --lora_path checkpoints/lora_audio2text_control_v1/best --n 6
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, load_dataset

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200
MAX_TEXT_TOKENS = 80


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


def _is_repeating(tokens: list[int], window: int = 8) -> bool:
    if len(tokens) < window * 2:
        return False
    return tokens[-window:] == tokens[-window * 2:-window]


def load_model_with_lora(lora_path: str | None):
    print(f"Loading model: {MODEL_ID}")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    model = LFM2AudioModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)

    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA: {lora_path}")
        from peft import LoraConfig, get_peft_model
        import safetensors.torch as st

        cfg = json.loads((Path(lora_path) / "adapter_config.json").read_text())
        lora_cfg = LoraConfig(
            r=cfg["r"], lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias=cfg.get("bias", "none"),
        )
        model = get_peft_model(model, lora_cfg)
        weights = st.load_file(str(Path(lora_path) / "adapter_model.safetensors"))
        model.load_state_dict(weights, strict=False)
        print("LoRA loaded.")

    model.eval()
    return model, processor


def load_pairs() -> dict[str, str]:
    pairs = {}
    with open("data/dialect_standard_pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs[normalize_text(obj["dialect"])] = obj["standard"]
    return pairs


def predict(model, processor, audio: np.ndarray) -> str:
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(SYSTEM_PROMPT)
    chat.end_turn()
    chat.new_turn("user")
    wav = torch.from_numpy(audio[None]).float().to(DEVICE)
    chat.add_audio(wav, 16000)
    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens: list[int] = []
    with torch.no_grad():
        for token in model.generate_interleaved(**chat, max_new_tokens=MAX_NEW_TOKENS, text_top_k=50):
            if token.numel() == 1:
                tid = token.item()
                if tid == 7:
                    break
                if tid == 130:
                    continue
                text_tokens.append(tid)
                if len(text_tokens) >= MAX_TEXT_TOKENS:
                    break
                if _is_repeating(text_tokens):
                    break
            else:
                if token[0].item() == 2048:
                    break

    if not text_tokens:
        return ""
    result = processor.text.decode(text_tokens, skip_special_tokens=True)
    return result if isinstance(result, str) else result[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="checkpoints/lora_audio2text_control_v1/best")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--output", default="eval_audio2text_control_results.jsonl")
    args = parser.parse_args()

    model, processor = load_model_with_lora(args.lora_path)
    pairs = load_pairs()

    # eval_audio2audio.py と完全に同一のtest分割ロジック (osaka only, seed=42, 先頭10%)
    print("Loading test split (osaka, same seed=42 split as audio2audio control)...")
    ds = load_dataset("federerjiang/dialect.osaka", split="train").shuffle(seed=42)
    n = len(ds)
    test_n = max(1, int(n * 0.1))
    ds_test = ds.select(range(test_n)).cast_column("audio", Audio(sampling_rate=16000))
    print(f"Test split: {len(ds_test)} samples, 評価: {args.n} samples")

    results = []
    cer_scores = []
    i = 0
    for sample in ds_test:
        if i >= args.n:
            break
        dialect_text = normalize_text(sample["sentence"])
        standard_ref = pairs.get(dialect_text, "")
        if not standard_ref:
            continue

        audio = sample["audio"]["array"].astype(np.float32)
        hyp = predict(model, processor, audio)
        score = cer(hyp, standard_ref)
        cer_scores.append(score)

        print(f"[{i+1}/{args.n}] CER={score:.3f}")
        print(f"  方言:   {dialect_text[:50]}")
        print(f"  予測:   {hyp[:50]}")
        print(f"  参照:   {standard_ref[:50]}")

        results.append({
            "sample": i + 1,
            "dialect": dialect_text,
            "hyp": hyp,
            "ref": standard_ref,
            "cer": score,
        })
        i += 1

    mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    print(f"\n=== 結果 (lora_audio2text_control_v1, 同一60件データ・同一6テストサンプル) ===")
    print(f"評価件数: {len(results)}")
    print(f"平均CER:  {mean_cer:.4f} ({mean_cer*100:.1f}%)")

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"結果保存: {out}")
    return mean_cer


if __name__ == "__main__":
    main()
