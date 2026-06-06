"""
Fine-tune Whisper large-v3 on the same dialect data as LFM for fair comparison.

Same conditions as LFM LoRA:
  - Data: federerjiang/dialect.osaka + dialect.kumamoto
  - Train samples: 260 (90% of 290), Test: 30 (same test split seed=42)
  - Method: LoRA r=16, alpha=32
  - Steps: 500, lr=1e-4

Usage:
    uv run python train_whisper_baseline.py
    uv run python train_whisper_baseline.py --model kotoba-tech/kotoba-whisper-v2.1 --output checkpoints/kotoba_lora
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jiwer
import torch
from datasets import Audio, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "openai/whisper-large-v3"
MAX_STEPS = 500
BATCH_SIZE = 4
LR = 1e-4
LORA_R = 16
LORA_ALPHA = 32


def load_dialect_splits(seed: int = 42):
    ds_o = load_dataset("federerjiang/dialect.osaka", split="train").cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    ds_k = load_dataset("federerjiang/dialect.kumamoto", split="train").cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    combined = concatenate_datasets([ds_o, ds_k]).shuffle(seed=seed)
    n = len(combined)
    test_n = max(1, int(n * 0.1))
    test_ds = combined.select(range(test_n))
    train_ds = combined.select(range(test_n, n))
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")
    return train_ds, test_ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: AutoProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [
            {"input_features": self.processor.feature_extractor(
                f["audio"]["array"], sampling_rate=16000
            ).input_features[0]}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [
            self.processor.tokenizer(f["sentence"]).input_ids
            for f in features
        ]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features}, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def compute_cer_fn(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = jiwer.cer(label_str, pred_str)
        return {"cer": cer}
    return compute_metrics


def evaluate_model(model, processor, test_ds, device, max_samples=None):
    model.eval()
    hyps, refs = [], []
    samples = test_ds.select(range(min(max_samples or len(test_ds), len(test_ds))))
    for sample in tqdm(samples, desc="Eval"):
        ref = sample["sentence"]
        arr = torch.tensor(sample["audio"]["array"]).float().unsqueeze(0)
        feats = processor.feature_extractor(
            sample["audio"]["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            pred_ids = model.generate(
                feats,
                language="japanese",
                task="transcribe",
                max_new_tokens=256,
            )
        hyp = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        hyps.append(hyp)
        refs.append(ref)
    cer = jiwer.cer(refs, hyps)
    examples = [{"ref": r, "hyp": h} for r, h in zip(refs[:5], hyps[:5])]
    return cer, examples


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model: {args.model}")
    print(f"Device: {device}")

    train_ds, test_ds = load_dialect_splits()

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language="japanese", task="transcribe")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Baseline CER before fine-tuning
    print("\n=== Baseline CER (before fine-tuning) ===")
    baseline_cer, baseline_examples = evaluate_model(model, processor, test_ds, device)
    print(f"Baseline CER: {baseline_cer:.4f} ({baseline_cer*100:.1f}%)")

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    output_dir = args.output
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        warmup_steps=max(1, MAX_STEPS // 10),
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=256,
        save_steps=100,
        eval_steps=50,
        logging_steps=10,
        report_to=["none"],
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        push_to_hub=False,
        dataloader_num_workers=4,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_cer_fn(processor),
        tokenizer=processor.feature_extractor,
    )

    print("\n=== Training ===")
    trainer.train()

    # Final CER after fine-tuning
    print("\n=== Fine-tuned CER ===")
    ft_cer, ft_examples = evaluate_model(model.merge_and_unload(), processor, test_ds, device)
    print(f"Fine-tuned CER: {ft_cer:.4f} ({ft_cer*100:.1f}%)")
    rel = (baseline_cer - ft_cer) / baseline_cer * 100
    print(f"Relative improvement: {rel:.1f}%")

    result = {
        "model": args.model,
        "lora_r": LORA_R,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "baseline_cer": baseline_cer,
        "finetuned_cer": ft_cer,
        "relative_improvement_pct": rel,
        "baseline_examples": baseline_examples,
        "finetuned_examples": ft_examples,
    }
    out_path = Path(output_dir) / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Results saved to {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="checkpoints/whisper_lora")
    args = parser.parse_args()
    main(args)
