"""
Test: lora_convert adapter + generate_interleaved → 方言音声 → 標準語音声

Usage:
    uv run python test_interleaved.py [--lora_path checkpoints/lora_convert/best] [--n 3]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias=cfg.get("bias", "none"),
        )
        model = get_peft_model(model, lora_cfg)
        weights = st.load_file(str(Path(lora_path) / "adapter_model.safetensors"))
        model.load_state_dict(weights, strict=False)
        print(f"LoRA loaded.")

    model.eval()
    return model, processor


def run_interleaved(model, processor, wav_array: np.ndarray, sr: int, max_new_tokens: int = 600):
    wav = torch.from_numpy(wav_array.T if wav_array.ndim > 1 else wav_array[None]).float()
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.to(DEVICE)

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(SYSTEM_PROMPT)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(wav, sr)
    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens: list[int] = []
    audio_frames: list[torch.Tensor] = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for token in model.generate_interleaved(**chat, max_new_tokens=max_new_tokens):
            if token.numel() == 1:
                tid = token.item()
                if tid == 7:  # <|im_end|>
                    break
                text_tokens.append(tid)
            else:
                # 音声フレーム (8 codebook values)
                if token[0].item() != 2048:  # EOA でなければ追加
                    audio_frames.append(token.cpu())

    elapsed = time.perf_counter() - t0

    # テキストデコード
    text = ""
    if text_tokens:
        result = processor.text.decode(text_tokens, skip_special_tokens=True)
        text = result if isinstance(result, str) else result[0]

    # 音声デコード
    waveform = None
    if audio_frames:
        # shape: (8, T) の整数テンソルにまとめる → (1, 8, T) で decode
        codes = torch.stack(audio_frames, dim=1)  # (8, T)
        codes = codes.unsqueeze(0).to(DEVICE)     # (1, 8, T)
        # EOA(2048)を含むフレームをクリップ
        codes = codes.clamp(0, 2047)
        with torch.no_grad():
            waveform = processor.decode(codes)    # (1, T')

    return text, waveform, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="checkpoints/lora_convert/best")
    parser.add_argument("--n", type=int, default=3, help="Number of samples to test")
    parser.add_argument("--max_tokens", type=int, default=250)
    args = parser.parse_args()

    model, processor = load_model_with_lora(args.lora_path)

    # 元データから音声をロード
    print("Loading dialect dataset...")
    ds_osaka = load_dataset("federerjiang/dialect.osaka", split="train", trust_remote_code=True)
    ds_osaka = ds_osaka.cast_column("audio", Audio(sampling_rate=16000))

    results = []
    for i in range(min(args.n, len(ds_osaka))):
        sample = ds_osaka[i]
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        ref_text = sample.get("sentence", sample.get("text", ""))

        print(f"\n[Sample {i+1}]")
        print(f"  方言テキスト(参考): {ref_text}")

        text, waveform, elapsed = run_interleaved(
            model, processor, audio_array, sr, max_new_tokens=args.max_tokens
        )

        print(f"  変換テキスト: {text}")
        print(f"  音声フレーム: {'あり' if waveform is not None else 'なし'}")
        if waveform is not None:
            duration = waveform.shape[-1] / 24000
            print(f"  音声長: {duration:.2f}s")
        print(f"  推論時間: {elapsed:.2f}s")

        # WAV保存
        if waveform is not None:
            out_path = f"/tmp/interleaved_output_{i+1}.wav"
            wav_np = waveform[0].float().cpu().numpy()
            sf.write(out_path, wav_np, 24000)
            print(f"  保存: {out_path}")

        results.append({
            "sample": i + 1,
            "dialect_ref": ref_text,
            "standard_text": text,
            "has_audio": waveform is not None,
            "audio_duration_s": waveform.shape[-1] / 24000 if waveform is not None else 0,
            "elapsed_s": elapsed,
        })

    print("\n=== 結果サマリー ===")
    for r in results:
        audio_info = f"{r['audio_duration_s']:.1f}s 音声" if r["has_audio"] else "音声なし"
        print(f"  [{r['sample']}] {r['dialect_ref'][:30]} → {r['standard_text'][:30]} ({audio_info})")


if __name__ == "__main__":
    main()
