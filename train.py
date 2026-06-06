"""
Fine-tune LFM2.5-Audio-1.5B-JP for elderly/dialect Japanese ASR.

Follows the official hackathon audio track recipe (scripts/audio/train.py)
and adds optional LoRA via PEFT for efficient personalization — our differentiator.

Usage:
    # 1. Preprocess data first:
    uv run python data/prepare_data.py --output data/cv_elderly_train
    uv run python data/prepare_data.py --split validation --output data/cv_elderly_val --max_samples 200

    # 2a. LoRA fine-tune (main method, ~15MB adapter):
    uv run python train.py

    # 2b. Full fine-tune (upper bound baseline):
    uv run python train.py --no_lora --output checkpoints/full

    # 3. Smoke test (5 steps):
    uv run python train.py --max_steps 5 --batch_size 1 --output checkpoints/smoke
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb
from peft import LoraConfig, get_peft_model
from liquid_audio import LFM2AudioModel
from liquid_audio.data.dataloader import LFM2DataLoader
from liquid_audio.trainer import Trainer
from liquid_audio.utils import get_model_dir

if TYPE_CHECKING:
    from liquid_audio.model.lfm2_audio import LFM2AudioModelOutput

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_ID      = os.environ.get("MODEL_ID",     "LiquidAI/LFM2.5-Audio-1.5B-JP")
MAX_STEPS     = int(os.environ.get("MAX_STEPS",    "1000"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",   "4"))
LR            = float(os.environ.get("LR",         "3e-5"))
OUTPUT_DIR    = Path(os.environ.get("OUTPUT_DIR",  "checkpoints/lora_r16"))
PUSH_TO_HUB   = os.environ.get("PUSH_TO_HUB")
USE_LORA      = os.environ.get("NO_LORA", "0") == "0"
LORA_R        = int(os.environ.get("LORA_R",      "16"))
LORA_ALPHA    = int(os.environ.get("LORA_ALPHA",   "32"))
TRAIN_DATA    = os.environ.get("TRAIN_DATA",   "data/dialect_train")
VAL_DATA      = os.environ.get("VAL_DATA",     "data/dialect_val")
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "512"))
WARMUP_STEPS  = max(1, MAX_STEPS // 10)

# LoRA targets: attention + FFN layers in LFM2 backbone
# Confirmed from probe: q_proj, k_proj, v_proj, out_proj (attn), w1, w2, w3 (FFN)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3"]


class LoRATrainer(Trainer):
    """liquid-audio Trainer + LoRA support + W&B logging.

    Overrides __init__ to apply LoRA before the optimizer is created,
    so only LoRA params are in the optimizer from the start.
    """

    # Signature must match Trainer.__init__; we add use_lora/lora_* kwargs.
    def __init__(
        self,
        model_id: str = MODEL_ID,
        train_data=None,
        val_data=None,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
        min_ratio: float = 0.1,
        max_steps: int = 500,
        warmup_steps: int = 50,
        batch_size: int = 4,
        dataloader_num_workers: int = 4,
        logging_interval: int = 10,
        save_interval: int = 100,
        val_interval: int = 50,
        output_dir: str = "checkpoints/lora",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        # Reproduce Trainer.__init__ exactly, inserting LoRA before optimizer creation
        import time
        from accelerate import Accelerator
        from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, ProjectConfiguration
        from torch.utils.data import DataLoader
        from liquid_audio.data.dataloader import lfm2_collator

        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.logging_interval = logging_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.output_dir = output_dir

        self.accelerator = Accelerator(
            mixed_precision="bf16",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
            dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
            project_config=ProjectConfiguration(
                project_dir=output_dir,
                automatic_checkpoint_naming=True,
                total_limit=5,
            ),
        )

        # Load base model
        self.accelerator.print(f"Loading model: {model_id}")
        model = LFM2AudioModel.from_pretrained(
            model_id, device=self.accelerator.device, dtype=torch.bfloat16
        )

        # Apply LoRA BEFORE optimizer creation
        if use_lora:
            lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=LORA_TARGETS,
                lora_dropout=0.05, bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            self.accelerator.print(
                f"LoRA applied: {trainable:,} / {total:,} params trainable ({100*trainable/total:.2f}%)"
            )
        else:
            self.accelerator.print("Full fine-tuning (no LoRA)")

        self.model = model

        # Optimizer only over trainable params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, betas=betas, eps=1e-8, weight_decay=weight_decay
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=lr * min_ratio
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        if train_data is None:
            raise ValueError("train_data is required")
        self.train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, collate_fn=lfm2_collator,
            num_workers=dataloader_num_workers, pin_memory=True,
            persistent_workers=dataloader_num_workers > 0,
            prefetch_factor=2 if dataloader_num_workers > 0 else None,
        )
        self.val_loader = None
        if val_data is not None:
            self.val_loader = DataLoader(
                val_data, batch_size=batch_size, shuffle=False, collate_fn=lfm2_collator,
                num_workers=dataloader_num_workers, pin_memory=True,
                persistent_workers=dataloader_num_workers > 0,
                prefetch_factor=2 if dataloader_num_workers > 0 else None,
            )

        (self.model, self.optimizer, self.train_loader,
         self.val_loader, self.scheduler) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        self.optimizer.zero_grad()
        self.step = 0
        self.epoch = 0
        self.time = 0.0
        self.best_val_loss = float("inf")

    def resume(self, checkpoint_dir: str, resume_step: int) -> None:
        """Resume training from an accelerate checkpoint."""
        self.accelerator.load_state(checkpoint_dir)
        self.step = resume_step
        self.accelerator.print(f"Resumed from {checkpoint_dir} at step {resume_step}")

    @torch.no_grad()
    def validate(self) -> None:
        if self.val_loader is None:
            return

        import time as _time
        loss_sum = torch.zeros(1, device=self.accelerator.device)
        loss_count = torch.zeros(1, device=self.accelerator.device)

        for batch in self.val_loader:
            batch = batch.to(self.accelerator.device)
            with self.accelerator.autocast():
                out = self.model(batch)
            loss_sum += out.loss.detach()
            loss_count += 1

        global_loss = (
            self.accelerator.reduce(loss_sum, reduction="sum")
            / self.accelerator.reduce(loss_count, reduction="sum").clamp_min(1)
        ).item()

        total = int(_time.monotonic() - self.time)
        mins, secs = divmod(total, 60)
        self.accelerator.print(
            f"[{mins:02d}:{secs:02d}] VALIDATION: step={self.step}/{self.max_steps} val_loss={global_loss:.4f}"
            + (" *** best ***" if global_loss < self.best_val_loss else "")
        )
        wandb.log({"val/loss": global_loss, "train/step": self.step}, step=self.step)

        if global_loss < self.best_val_loss:
            self.best_val_loss = global_loss
            self.accelerator.save_model(
                self.accelerator.unwrap_model(self.model),
                f"{self.output_dir}/best",
                max_shard_size="5GB",
                safe_serialization=True,
            )
            self.accelerator.print(f"  -> Best model saved to {self.output_dir}/best")

    def log(self, model_output: LFM2AudioModelOutput) -> None:
        super().log(model_output)
        if self.step > 0 and self.step % self.logging_interval == 0:
            wandb.log(
                {
                    "train/loss": model_output.loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/step": self.step,
                },
                step=self.step,
            )


def assemble_checkpoint(final_dir: Path, model_id: str, use_lora: bool) -> None:
    """Make final_dir a self-contained loadable checkpoint.

    For LoRA: extracts adapter weights into PEFT-format (~44MB).
    For full fine-tune: copies config/tokenizer alongside safetensors.
    """
    import json as _json
    import safetensors.torch as _st

    if not final_dir.exists():
        raise RuntimeError(f"No checkpoint at {final_dir}")

    if use_lora:
        # Extract LoRA keys from full-model safetensors and save as PEFT adapter
        full_sd = _st.load_file(str(final_dir / "model.safetensors"))
        lora_sd = {k: v for k, v in full_sd.items() if "lora" in k.lower()}
        if not lora_sd:
            print("Warning: no LoRA keys found in checkpoint!")
        else:
            _st.save_file(lora_sd, str(final_dir / "adapter_model.safetensors"))
            adapter_config = {
                "base_model_name_or_path": model_id,
                "bias": "none",
                "fan_in_fan_out": False,
                "inference_mode": True,
                "init_lora_weights": True,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "r": LORA_R,
                "target_modules": LORA_TARGETS,
            }
            with open(final_dir / "adapter_config.json", "w") as f:
                _json.dump(adapter_config, f, indent=2)
            print(f"LoRA adapter saved: {len(lora_sd)} keys, {sum(v.numel()*v.element_size() for v in lora_sd.values())/1e6:.1f} MB")
    else:
        # Copy config/tokenizer from base model alongside safetensors
        base_dir = get_model_dir(model_id)
        for entry in base_dir.iterdir():
            if entry.name == "model.safetensors" or entry.name.startswith("."):
                continue
            if entry.is_dir():
                shutil.copytree(entry, final_dir / entry.name, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, final_dir / entry.name)
    print(f"Checkpoint ready at: {final_dir}")


def push_to_hub(folder: Path, repo_id: str) -> str:
    from huggingface_hub import HfApi
    print(f"Pushing to https://huggingface.co/{repo_id} ...")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        commit_message="elderly-asr lora fine-tune",
    )
    print(f"Push complete. Hub revision: {commit.oid}")
    return commit.oid


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training.")

    wandb_mode = "disabled" if not os.environ.get("WANDB_API_KEY") else os.environ.get("WANDB_MODE", "online")
    with wandb.init(
        project=os.environ.get("WANDB_PROJECT", "hack-liquid-way"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=os.environ.get("WANDB_RUN_NAME", f"lfm-elderly-asr-{'lora' if USE_LORA else 'full'}"),
        tags=["audio", "asr", "elderly", "lfm2.5-audio-1.5b-jp", "lora" if USE_LORA else "full"],
        config={
            "model": MODEL_ID,
            "use_lora": USE_LORA,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
        },
        mode=wandb_mode,
    ) as run:
        print(f"W&B run: {run.url if wandb_mode != 'disabled' else '(W&B disabled)'}")

        train_data = LFM2DataLoader(TRAIN_DATA, context_length=CONTEXT_LENGTH)
        val_data = LFM2DataLoader(VAL_DATA, context_length=CONTEXT_LENGTH) if Path(VAL_DATA).exists() else None

        trainer = LoRATrainer(
            model_id=MODEL_ID,
            train_data=train_data,
            val_data=val_data,
            lr=LR,
            batch_size=BATCH_SIZE,
            max_steps=MAX_STEPS,
            warmup_steps=WARMUP_STEPS,
            dataloader_num_workers=4,
            logging_interval=max(1, MAX_STEPS // 50),
            save_interval=max(50, MAX_STEPS // 5),
            val_interval=max(25, MAX_STEPS // 10),
            output_dir=str(OUTPUT_DIR),
            use_lora=USE_LORA,
            lora_r=LORA_R,
            lora_alpha=LORA_ALPHA,
        )

        resume_from = os.environ.get("RESUME_FROM")
        resume_step = int(os.environ.get("RESUME_STEP", "0"))
        if resume_from:
            trainer.resume(resume_from, resume_step)

        trainer.train()

        assemble_checkpoint(OUTPUT_DIR / "final", MODEL_ID, USE_LORA)

        # Also assemble best checkpoint if it exists (val_loss が最低のステップ)
        best_dir = OUTPUT_DIR / "best"
        if best_dir.exists() and (best_dir / "model.safetensors").exists():
            assemble_checkpoint(best_dir, MODEL_ID, USE_LORA)
            wandb.run.summary["best_val_loss"] = trainer.best_val_loss
            print(f"Best val_loss={trainer.best_val_loss:.4f} saved at {best_dir}")

        if PUSH_TO_HUB:
            oid = push_to_hub(OUTPUT_DIR / "best" if best_dir.exists() else OUTPUT_DIR / "final", PUSH_TO_HUB)
            wandb.run.summary["hf_repo"] = PUSH_TO_HUB
            wandb.run.summary["hf_revision"] = oid

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # CLI overrides env vars
    if args.no_lora:
        os.environ["NO_LORA"] = "1"
    if args.lora_r is not None:
        os.environ["LORA_R"] = str(args.lora_r)
    if args.lora_alpha is not None:
        os.environ["LORA_ALPHA"] = str(args.lora_alpha)
    if args.max_steps is not None:
        os.environ["MAX_STEPS"] = str(args.max_steps)
    if args.batch_size is not None:
        os.environ["BATCH_SIZE"] = str(args.batch_size)
    if args.lr is not None:
        os.environ["LR"] = str(args.lr)
    if args.output is not None:
        os.environ["OUTPUT_DIR"] = args.output

    # Re-read env after overrides
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "1000"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
    LR = float(os.environ.get("LR", "3e-5"))
    OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "checkpoints/lora_r16"))
    USE_LORA = os.environ.get("NO_LORA", "0") == "0"
    LORA_R = int(os.environ.get("LORA_R", "16"))
    LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
    WARMUP_STEPS = max(1, MAX_STEPS // 10)

    main()
