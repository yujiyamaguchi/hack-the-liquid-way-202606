"""
Evaluate ASR performance (CER) comparing baseline vs fine-tuned LFM2.5-Audio-1.5B-JP.

Usage:
    # Baseline only:
    uv run python evaluate.py --test_data data/cv_elderly_test

    # Baseline + fine-tuned comparison:
    uv run python evaluate.py --test_data data/cv_elderly_test --lora_path checkpoints/lora_r16/final

    # Save results to JSON:
    uv run python evaluate.py --test_data data/cv_elderly_test --lora_path checkpoints/lora_r16/final --output results.json
"""
import argparse
import json
from pathlib import Path

import jiwer
import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
ASR_SYSTEM_PROMPT = "Perform ASR in japanese."


def transcribe(model: LFM2AudioModel, processor: LFM2AudioProcessor, wav: torch.Tensor, sampling_rate: int) -> str:
    """Run ASR on a single audio clip."""
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(ASR_SYSTEM_PROMPT)
    chat.end_turn()

    chat.new_turn("user")
    chat.add_audio(wav, sampling_rate)
    chat.end_turn()

    chat.new_turn("assistant")

    text_tokens: list[torch.Tensor] = []
    with torch.no_grad():
        for t in model.generate_sequential(**chat, max_new_tokens=256):
            if t.numel() == 1:
                text_tokens.append(t)

    if not text_tokens:
        return ""
    token_ids = [t.item() for t in text_tokens]
    result = processor.text.decode(token_ids, skip_special_tokens=True)
    return result if isinstance(result, str) else result[0]


def compute_cer(hypotheses: list[str], references: list[str]) -> float:
    """Compute character error rate for Japanese."""
    return jiwer.cer(references, hypotheses)


def evaluate_model(
    model: LFM2AudioModel,
    processor: LFM2AudioProcessor,
    test_dataset,
    max_samples: int | None = None,
    device: str = "cuda",
) -> dict:
    model.eval()
    hypotheses = []
    references = []

    samples = test_dataset
    if max_samples:
        samples = test_dataset.select(range(min(max_samples, len(test_dataset))))

    for sample in tqdm(samples, desc="Evaluating"):
        ref = sample["sentence"]
        if not ref:
            continue

        audio_info = sample["audio"]
        wav = torch.from_numpy(audio_info["array"]).float().unsqueeze(0).to(device)
        sr = audio_info["sampling_rate"]

        try:
            hyp = transcribe(model, processor, wav, sr)
        except Exception as e:
            print(f"  Warning: transcription failed: {e}")
            hyp = ""

        hypotheses.append(hyp)
        references.append(ref)

    cer = compute_cer(hypotheses, references)
    return {
        "cer": cer,
        "num_samples": len(references),
        "examples": [
            {"ref": r, "hyp": h} for r, h in zip(references[:5], hypotheses[:5])
        ],
    }


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from datasets import load_dataset, Audio, concatenate_datasets
    if args.dialect:
        print("Loading dialect test data...")
        datasets_list = []
        for name in ("osaka", "kumamoto"):
            ds = load_dataset(f"federerjiang/dialect.{name}", split="train")
            datasets_list.append(ds)
        combined = concatenate_datasets(datasets_list).shuffle(seed=42)
        n = len(combined)
        test_n = max(1, int(n * 0.1))
        test_ds = combined.select(range(test_n))
        print(f"Dialect test samples: {len(test_ds)}")
    else:
        print(f"Loading test data from {args.test_data}...")
        test_ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "ja",
            split="test",
        )
        if not args.all_ages:
            elderly_ages = {"sixties", "seventies", "eighties", "nineties"}
            test_ds = test_ds.filter(lambda x: x.get("age") in elderly_ages)
            print(f"Elderly test samples: {len(test_ds)}")

    print(f"Loading base model: {MODEL_ID}")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    base_model = LFM2AudioModel.from_pretrained(MODEL_ID, device=device, dtype=torch.bfloat16)

    print("\n=== Baseline evaluation ===")
    baseline_results = evaluate_model(base_model, processor, test_ds, max_samples=args.max_samples, device=device)
    print(f"Baseline CER: {baseline_results['cer']:.4f} ({baseline_results['cer']*100:.1f}%)")
    print("Examples:")
    for ex in baseline_results["examples"]:
        print(f"  REF: {ex['ref']}")
        print(f"  HYP: {ex['hyp']}")
        print()

    results = {"baseline": baseline_results}

    if args.lora_path:
        print(f"\n=== Fine-tuned evaluation ({args.lora_path}) ===")
        from peft import LoraConfig, get_peft_model
        import safetensors.torch as st_load
        import json as _json
        adapter_cfg_path = Path(args.lora_path) / "adapter_config.json"
        adapter_cfg = _json.loads(adapter_cfg_path.read_text())
        lora_cfg = LoraConfig(
            r=adapter_cfg["r"], lora_alpha=adapter_cfg["lora_alpha"],
            target_modules=adapter_cfg["target_modules"],
            lora_dropout=adapter_cfg.get("lora_dropout", 0.05), bias=adapter_cfg.get("bias", "none"),
        )
        finetuned_model = get_peft_model(base_model, lora_cfg)
        adapter_weights = st_load.load_file(str(Path(args.lora_path) / "adapter_model.safetensors"))
        finetuned_model.load_state_dict(adapter_weights, strict=False)
        finetuned_model.eval()
        finetuned_results = evaluate_model(finetuned_model, processor, test_ds, max_samples=args.max_samples, device=device)
        print(f"Fine-tuned CER: {finetuned_results['cer']:.4f} ({finetuned_results['cer']*100:.1f}%)")
        rel_improvement = (baseline_results["cer"] - finetuned_results["cer"]) / baseline_results["cer"] * 100
        print(f"Relative CER improvement: {rel_improvement:.1f}%")
        print("Examples:")
        for ex in finetuned_results["examples"]:
            print(f"  REF: {ex['ref']}")
            print(f"  HYP: {ex['hyp']}")
            print()
        results["finetuned"] = finetuned_results
        results["relative_improvement_pct"] = rel_improvement

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR: baseline vs fine-tuned")
    parser.add_argument("--test_data", default="data/cv_elderly_test", help="Processed test dataset path")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--output", default=None, help="Save results to JSON")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--all_ages", action="store_true", help="Evaluate on all ages, not just elderly")
    parser.add_argument("--dialect", action="store_true", help="Evaluate on dialect test set (osaka+kumamoto)")
    args = parser.parse_args()

    main(args)
