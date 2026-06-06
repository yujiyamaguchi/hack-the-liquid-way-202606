"""
方言→標準語変換精度評価スクリプト
案①（テキスト入力）・案②（音声入力）両対応

Usage:
    # 案②: 音声入力 (generate_interleaved)
    uv run python eval_convert.py --lora_path checkpoints/lora_convert_v4/best --input_mode audio

    # 案①: テキスト入力
    uv run python eval_convert.py --lora_path checkpoints/lora_text_v1/best --input_mode text

    # ベースモデル（LoRAなし）
    uv run python eval_convert.py --input_mode audio
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, concatenate_datasets, load_dataset

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT_AUDIO = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
SYSTEM_PROMPT_TEXT  = "次の方言テキストを自然な標準語（です・ます調）に変換してください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 300
MAX_AUDIO_FRAMES = 200
MAX_TEXT_TOKENS = 80


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate（文字誤り率）"""
    h = list(hyp.replace(" ", ""))
    r = list(ref.replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    # 動的計画法でレーベンシュタイン距離計算
    dp = list(range(len(r) + 1))
    for i, ch in enumerate(h):
        ndp = [i + 1] + [0] * len(r)
        for j, cr in enumerate(r):
            ndp[j + 1] = min(dp[j] + (0 if ch == cr else 1), dp[j + 1] + 1, ndp[j] + 1)
        dp = ndp
    return dp[-1] / len(r)


def load_model(lora_path: str | None):
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


def predict(model, processor, audio: np.ndarray | None, dialect_text: str, input_mode: str) -> str:
    """方言音声またはテキスト → 標準語テキストを予測"""
    system_prompt = SYSTEM_PROMPT_TEXT if input_mode == "text" else SYSTEM_PROMPT_AUDIO

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(system_prompt)
    chat.end_turn()
    chat.new_turn("user")

    if input_mode == "text":
        chat.add_text(dialect_text)
    else:
        wav = torch.from_numpy(audio[None]).float().to(DEVICE)
        chat.add_audio(wav, 16000)

    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens: list[int] = []
    audio_frames = 0

    def _is_repeating(tokens: list[int], window: int = 8) -> bool:
        if len(tokens) < window * 2:
            return False
        return tokens[-window:] == tokens[-window * 2:-window]

    with torch.no_grad():
        for token in model.generate_interleaved(
            **chat, max_new_tokens=MAX_NEW_TOKENS, text_top_k=50,
        ):
            if token.numel() == 1:
                tid = token.item()
                if tid == 7:   # <|im_end|>
                    break
                if tid == 130: # <|text_end|>
                    continue
                text_tokens.append(tid)
                if len(text_tokens) >= MAX_TEXT_TOKENS:
                    break
                if _is_repeating(text_tokens):
                    break
            else:
                if token[0].item() == 2048:  # EOA
                    break
                audio_frames += 1
                if audio_frames >= MAX_AUDIO_FRAMES:
                    break

    if not text_tokens:
        return ""
    result = processor.text.decode(text_tokens, skip_special_tokens=True)
    return result if isinstance(result, str) else result[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--input_mode", default="audio", choices=["audio", "text"])
    parser.add_argument("--n", type=int, default=50, help="評価サンプル数")
    parser.add_argument("--output", default=None, help="結果JSONLの保存先")
    args = parser.parse_args()

    model, processor = load_model(args.lora_path)
    pairs = load_pairs()

    # test splitをロード（train/val/testと同じseed=42分割）
    print("Loading test split...")
    ds_osaka = load_dataset("federerjiang/dialect.osaka", split="train")
    ds_kuma  = load_dataset("federerjiang/dialect.kumamoto", split="train")
    combined = concatenate_datasets([ds_osaka, ds_kuma]).shuffle(seed=42)
    n = len(combined)
    test_n = max(1, int(n * 0.1))
    ds_test = combined.select(range(test_n))
    if args.input_mode == "audio":
        ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Test split: {len(ds_test)} samples, 評価: {args.n} samples")

    results = []
    cer_scores = []

    for i, sample in enumerate(ds_test):
        if i >= args.n:
            break

        dialect_text = normalize_text(sample["sentence"])
        standard_ref = pairs.get(dialect_text, "")
        if not standard_ref:
            continue

        if args.input_mode == "audio":
            audio = sample["audio"]["array"].astype(np.float32)
        else:
            audio = None

        hyp = predict(model, processor, audio, dialect_text, args.input_mode)
        score = cer(hyp, standard_ref)
        cer_scores.append(score)

        print(f"[{i+1}/{args.n}] CER={score:.3f}")
        print(f"  方言:   {dialect_text[:50]}")
        print(f"  予測:   {hyp[:50]}")
        print(f"  参照:   {standard_ref[:50]}")

        results.append({
            "dialect": dialect_text,
            "hyp": hyp,
            "ref": standard_ref,
            "cer": score,
        })

    mean_cer = np.mean(cer_scores) if cer_scores else 0.0
    print(f"\n=== 結果 ===")
    print(f"評価件数: {len(cer_scores)}")
    print(f"平均CER:  {mean_cer:.4f} ({mean_cer*100:.1f}%)")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"結果保存: {out}")

    return mean_cer


if __name__ == "__main__":
    main()
