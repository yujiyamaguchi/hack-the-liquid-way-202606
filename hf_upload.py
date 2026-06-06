"""
HuggingFace Hub アップロードスクリプト
- LoRA adapter  → YujiYamaguchi/lfm25-audio-jp-dialect-lora
- 学習データセット → YujiYamaguchi/dialect-to-standard-ja
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
api = HfApi(token=TOKEN)
HF_USER = "YujiYamaguchi"

# ─── モデルカード ───
MODEL_README = """\
---
language:
- ja
license: mit
base_model: LiquidAI/LFM2.5-Audio-1.5B-JP
tags:
- audio
- speech
- dialect
- lora
- peft
- japanese
datasets:
- YujiYamaguchi/dialect-to-standard-ja
---

# lfm25-audio-jp-dialect-lora

**Hack the Liquid WAY 2026 — Track 2 / チーム山口**

[LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) を LoRA fine-tune した、
方言音声（関西弁・熊本弁）→ 標準語テキスト変換アダプタです。

## 性能

| モデル | val_loss | 備考 |
|--------|----------|------|
| lora_convert_v4 (LoRA, 1000 steps) | **0.934** | **このアダプタ** |
| full_ft_v1 (Full FT, 800 steps) | 0.908 | 参考値（不採用）|

平均CER: 0.40（testセット 50件）/ 推論 1〜2秒（GPU）

## LoRA 設定

- r=16, alpha=32, dropout=0.05
- 対象層: q_proj / k_proj / v_proj / out_proj / w1 / w2 / w3（FFN）
- 訓練パラメータ: 11M / 1,464M = 0.76%
- 学習環境: RTX 5090

## 使い方

```python
import torch
import json
from pathlib import Path
from peft import LoraConfig, get_peft_model
import safetensors.torch as st
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from huggingface_hub import snapshot_download

# アダプタのダウンロード
adapter_dir = snapshot_download("YujiYamaguchi/lfm25-audio-jp-dialect-lora")

# モデルロード
MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
model = LFM2AudioModel.from_pretrained(MODEL_ID, device="cuda", dtype=torch.bfloat16)

# LoRA 適用
cfg = json.loads(Path(adapter_dir, "adapter_config.json").read_text())
lora_cfg = LoraConfig(
    r=cfg["r"], lora_alpha=cfg["lora_alpha"],
    target_modules=cfg["target_modules"],
    lora_dropout=cfg.get("lora_dropout", 0.05),
)
model = get_peft_model(model, lora_cfg)
weights = st.load_file(str(Path(adapter_dir, "adapter_model.safetensors")))
model.load_state_dict(weights, strict=False)
model.eval()
```

詳細は [GitHubリポジトリ](https://github.com/YujiYamaguchi/hack-the-liquid-way-202606) を参照。

## 学習データ

- 出典: [`federerjiang/dialect.osaka`](https://huggingface.co/datasets/federerjiang/dialect.osaka) +
  [`federerjiang/dialect.kumamoto`](https://huggingface.co/datasets/federerjiang/dialect.kumamoto)
- 加工: Qwen3-32B で方言テキスト → 標準語テキストを自動生成
- 学習ペア: 方言音声 → 標準語テキスト（大阪弁 + 熊本弁 計2,600件）
"""

# ─── データセットカード ───
DATASET_README = """\
---
language:
- ja
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
tags:
- dialect
- japanese
- osaka
- kumamoto
- speech-to-text
---

# dialect-to-standard-ja

方言音声（関西弁・熊本弁）→ 標準語テキスト変換のための学習データセット。

**Hack the Liquid WAY 2026 — Track 2 / チーム山口**

## データ構成

- 大阪弁 約1,300件 + 熊本弁 約1,300件 = **計2,600件**
- 分割: train 80% / val 10% / test 10%（seed=42）
- 形式: HuggingFace Datasets（Arrow）

### カラム

| カラム | 説明 |
|--------|------|
| `audio` | 方言音声（WAV 16kHz） |
| `dialect_text` | 方言テキスト（元データのラベル） |
| `standard_text` | 標準語テキスト（Qwen3-32Bで自動生成） |
| `source` | `osaka` or `kumamoto` |

### テキストペア（JSONL）

`dialect_standard_pairs.jsonl` に全2,600件のテキストペアを収録:

```json
{"dialect": "最近，下宿し始めたからちゃうか．", "standard": "最近，下宿し始めたからですか。", "source": "osaka"}
```

## 元データ出典

- [`federerjiang/dialect.osaka`](https://huggingface.co/datasets/federerjiang/dialect.osaka)
- [`federerjiang/dialect.kumamoto`](https://huggingface.co/datasets/federerjiang/dialect.kumamoto)

標準語テキストは Qwen3-32B（`data/generate_standard_pairs.py`）で自動生成しました。

## 関連リソース

- LoRA アダプタ: [`YujiYamaguchi/lfm25-audio-jp-dialect-lora`](https://huggingface.co/YujiYamaguchi/lfm25-audio-jp-dialect-lora)
- コード: [GitHub](https://github.com/YujiYamaguchi/hack-the-liquid-way-202606)
"""


def upload_model():
    repo_id = f"{HF_USER}/lfm25-audio-jp-dialect-lora"
    print(f"\n=== モデルリポジトリ作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    # モデルカード
    api.upload_file(
        path_or_fileobj=MODEL_README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  README.md アップロード完了")

    # adapter_config.json
    api.upload_file(
        path_or_fileobj="checkpoints/lora_convert_v4/best/adapter_config.json",
        path_in_repo="adapter_config.json",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  adapter_config.json アップロード完了")

    # adapter_model.safetensors (43MB)
    print("  adapter_model.safetensors アップロード中（43MB）...")
    api.upload_file(
        path_or_fileobj="checkpoints/lora_convert_v4/best/adapter_model.safetensors",
        path_in_repo="adapter_model.safetensors",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  完了！ https://huggingface.co/{repo_id}")
    return repo_id


def upload_dataset():
    repo_id = f"{HF_USER}/dialect-to-standard-ja"
    print(f"\n=== データセットリポジトリ作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    # データセットカード
    api.upload_file(
        path_or_fileobj=DATASET_README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  README.md アップロード完了")

    # テキストペア JSONL
    api.upload_file(
        path_or_fileobj="data/dialect_standard_pairs.jsonl",
        path_in_repo="dialect_standard_pairs.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  dialect_standard_pairs.jsonl アップロード完了")

    # Arrow データ（train / val / test）
    for split in ("train", "val", "test"):
        folder = f"data/dialect_convert_{split}"
        if Path(folder).exists():
            print(f"  {split}データ アップロード中...")
            api.upload_folder(
                folder_path=folder,
                path_in_repo=split,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"  {split} 完了")

    print(f"  完了！ https://huggingface.co/datasets/{repo_id}")
    return repo_id


if __name__ == "__main__":
    model_repo = upload_model()
    dataset_repo = upload_dataset()
    print(f"\n--- アップロード完了 ---")
    print(f"モデル  : https://huggingface.co/{model_repo}")
    print(f"データ  : https://huggingface.co/datasets/{dataset_repo}")
