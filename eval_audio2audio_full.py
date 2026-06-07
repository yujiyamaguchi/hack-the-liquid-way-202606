"""
方言音声 → 標準語「音声」変換 評価 (lora_audio2audio_full_v1 用)
lora_convert_v4 と全く同じデータセット(osaka+kumamoto, 2077/260/258)で学習したモデルを評価する。

eval_audio2audio.py をベースに、テスト分割を eval_convert.py と同じ
combined osaka+kumamoto / shuffle(seed=42) に変更。

Usage:
    uv run python eval_audio2audio_full.py --lora_path checkpoints/lora_audio2audio_full_v1/best --n 12
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, concatenate_datasets, load_dataset

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 200
MAX_TEXT_TOKENS = 80
SAMPLE_RATE = 24000


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


def predict(model, processor, audio: np.ndarray, max_new_tokens: int = MAX_NEW_TOKENS):
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
    audio_frames: list[torch.Tensor] = []

    import time
    t0 = time.perf_counter()
    with torch.no_grad():
        for token in model.generate_interleaved(**chat, max_new_tokens=max_new_tokens, text_top_k=50):
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
                audio_frames.append(token.cpu())
    elapsed = time.perf_counter() - t0

    text = ""
    if text_tokens:
        result = processor.text.decode(text_tokens, skip_special_tokens=True)
        text = result if isinstance(result, str) else result[0]

    waveform = None
    if audio_frames:
        codes = torch.stack(audio_frames, dim=1).unsqueeze(0).to(DEVICE)
        codes = codes.clamp(0, 2047)
        with torch.no_grad():
            waveform = processor.decode(codes)

    return text, waveform, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="checkpoints/lora_audio2audio_full_v1/best")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--max_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output_dir", default="eval_audio2audio_full_results")
    args = parser.parse_args()

    model, processor = load_model_with_lora(args.lora_path)
    pairs = load_pairs()

    # lora_convert_v4 / eval_convert.py と同じ test split (osaka+kumamoto combined, seed=42)
    print("Loading test split (osaka+kumamoto combined, same seed=42 split as lora_convert_v4)...")
    ds_osaka = load_dataset("federerjiang/dialect.osaka", split="train")
    ds_kuma = load_dataset("federerjiang/dialect.kumamoto", split="train")
    combined = concatenate_datasets([ds_osaka, ds_kuma]).shuffle(seed=42)
    n = len(combined)
    test_n = max(1, int(n * 0.1))
    ds_test = combined.select(range(test_n)).cast_column("audio", Audio(sampling_rate=16000))
    print(f"Test split: {len(ds_test)} samples, 評価: {args.n} samples")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
        text, waveform, elapsed = predict(model, processor, audio, max_new_tokens=args.max_tokens)
        score = cer(text, standard_ref)
        cer_scores.append(score)

        wav_path = None
        duration = 0.0
        if waveform is not None:
            wav_path = str(out_dir / f"sample_{i+1}.wav")
            wav_np = waveform[0].float().cpu().numpy()
            sf.write(wav_path, wav_np, SAMPLE_RATE)
            duration = wav_np.shape[-1] / SAMPLE_RATE

        print(f"[{i+1}/{args.n}] CER={score:.3f}  音声={'あり ' + f'{duration:.1f}s' if waveform is not None else 'なし'}  ({elapsed:.1f}s)")
        print(f"  方言:   {dialect_text[:50]}")
        print(f"  予測:   {text[:50]}")
        print(f"  参照:   {standard_ref[:50]}")
        if wav_path:
            print(f"  保存:   {wav_path}")

        results.append({
            "sample": i + 1,
            "dialect": dialect_text,
            "hyp_text": text,
            "ref_text": standard_ref,
            "cer": score,
            "wav_path": wav_path,
            "audio_duration_s": duration,
            "elapsed_s": elapsed,
        })
        i += 1

    mean_cer = float(np.mean(cer_scores)) if cer_scores else 0.0
    n_with_audio = sum(1 for r in results if r["wav_path"])
    print(f"\n=== 結果 (lora_audio2audio_full_v1, lora_convert_v4と同一データ・同一split) ===")
    print(f"評価件数:       {len(results)}")
    print(f"平均CER:        {mean_cer:.4f} ({mean_cer*100:.1f}%)")
    print(f"音声生成成功数: {n_with_audio}/{len(results)}")

    out_jsonl = out_dir / "results_audio2audio_full.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"結果保存: {out_jsonl}")
    return mean_cer


if __name__ == "__main__":
    main()
