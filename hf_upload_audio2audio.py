"""
HuggingFace Hub アップロードスクリプト（音声→音声 直接生成モデル / audio2audio）
- LoRA adapter  → YujiYamaguchi/lfm25-audio-jp-dialect-audio2audio-lora
- 学習データセット → YujiYamaguchi/dialect-to-standard-ja-speech

既存の hf_upload.py（lora_convert_v4: 方言音声→標準語テキスト）とは別モデル・別データセットとして
新規リポジトリにアップロードする（テキストのみ出力 vs テキスト+音声同時出力で出力形式が根本的に異なるため）。
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
- speech-to-speech
- dialect
- lora
- peft
- japanese
datasets:
- YujiYamaguchi/dialect-to-standard-ja-speech
---

# lfm25-audio-jp-dialect-audio2audio-lora

**Hack the Liquid WAY 2026 — Track 2 / チーム山口**

[LFM2.5-Audio-1.5B-JP](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-JP) を LoRA fine-tune した、
方言音声（関西弁・熊本弁）→ 標準語「テキスト＋音声」を**同時に直接生成**するアダプタです。

[`lfm25-audio-jp-dialect-lora`](https://huggingface.co/YujiYamaguchi/lfm25-audio-jp-dialect-lora)
（方言音声→標準語テキストのみ）の発展版で、こちらは ASR + LLM + TTS の3段構成を介さず、
**1モデル・1推論**で方言音声から標準語の音声をそのまま生成します（外部TTSエンジン不使用、
モデル自身の声で出力）。

## 性能（lora_convert_v4 と全く同一のデータ・同一レシピで学習したコントロール実験）

学習objectiveが異なる（テキストのみ vs テキスト+8系統のMimiオーディオコードブック）ため
val_lossの絶対値は直接比較できないが、参考として:

| モデル | 出力形式 | val_loss | テキスト成分CER |
|--------|---------|----------|----------------|
| lora_convert_v4 (既存・採用) | テキストのみ | 0.934（収束） | 0.40 |
| **lora_audio2audio_full_v1 (このアダプタ)** | **テキスト＋音声を同時生成** | 1.6966 → 1.3572（単調減少、改善継続中） | **0.362** |

→ 音声出力を同時に行っても、より複雑な複合タスクで安定して収束し、テキスト精度も劣化しない。

**生成音声の品質を独立ASRで定量検証**（自前のモデルで判定すると測定方法自体に偏りが出るため、
学習に一切使用していない第三者ASR `kotoba-whisper-v2.1` で検証）:
- クリーン参照音声での健全性チェック: CER 4.2% → 判定者として信頼できることを確認
- 生成音声の書き起こし: CER 37.7%（テキスト成分CER 36.2%とほぼ同水準）
  → 生成音声は「意味の通る自然な標準語日本語」として機能している（崩壊した音声ではない）

## LoRA 設定

- r=16, alpha=32, dropout=0.05
- 対象層: q_proj / k_proj / v_proj / out_proj / w1 / w2 / w3（FFN）
- 訓練パラメータ: 0.76%
- 学習環境: RTX 5090 / MAX_STEPS=1000, LR=1e-4

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
adapter_dir = snapshot_download("YujiYamaguchi/lfm25-audio-jp-dialect-audio2audio-lora")

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

# 推論: generate_interleaved でテキストと音声を交互に同時生成
# (詳細は eval_audio2audio_full.py の predict() を参照)
```

詳細は [GitHubリポジトリ](https://github.com/YujiYamaguchi/hack-the-liquid-way-202606) を参照。

## 学習データ・手法

- 出典: [`federerjiang/dialect.osaka`](https://huggingface.co/datasets/federerjiang/dialect.osaka) +
  [`federerjiang/dialect.kumamoto`](https://huggingface.co/datasets/federerjiang/dialect.kumamoto)
- 標準語テキストは Qwen3-32B で自動生成し、それを edge-tts で音声合成。各ペアを
  `InterleavedSegment`（テキストと音声フレームを交互配置する専用データ型）として
  `generate_interleaved` の出力パターンと一致する形式で学習データに格納（鍵となる工夫）
- 学習ペア: 方言音声 → 標準語テキスト＋音声（大阪弁 + 熊本弁 計2,600件、train/val/test = 2077/260/258）
- データセット: [`YujiYamaguchi/dialect-to-standard-ja-speech`](https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja-speech)
"""

# ─── データセットカード ───
DATASET_README = """\
---
language:
- ja
license: cc-by-4.0
task_categories:
- audio-to-audio
- automatic-speech-recognition
tags:
- dialect
- japanese
- osaka
- kumamoto
- speech-to-speech
- interleaved
---

# dialect-to-standard-ja-speech

方言音声（関西弁・熊本弁）→ 標準語「テキスト＋音声」を同時生成するモデルの学習データセット。

**Hack the Liquid WAY 2026 — Track 2 / チーム山口**

[`dialect-to-standard-ja`](https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja)
（方言音声→標準語テキストのみ）の発展版。各ペアの標準語ラベルに、テキストに加えて
edge-tts で合成した標準語音声を持たせ、`InterleavedSegment`
（テキストと音声フレームを交互配置する専用データ型）として
[liquid-audio](https://github.com/Liquid4All/liquid-audio) の `generate_interleaved` の
出力パターンと一致する形式に前処理済み（モデル学習にそのまま投入できる Arrow 形式、
LFM2.5-Audio-1.5B-JP のトークナイザ・Mimi コーデックでエンコード済み）。

## データ構成

- 大阪弁 + 熊本弁 = **計2,600件**
- 分割: train 2,077 / val 260 / test 258（seed=42、`dialect-to-standard-ja` と同一split）
- 形式: HuggingFace Datasets（Arrow、前処理済みテンソル）

### カラム

| カラム | 説明 |
|--------|------|
| `text` | トークンID列（システムプロンプト＋方言テキスト＋標準語テキストを含む対話） |
| `audio_in` | 入力（方言）音声の特徴量 |
| `audio_in_lens` | 入力音声の長さ |
| `audio_out` | 出力（標準語）音声の Mimi コードブック列（8系統、`InterleavedSegment` でテキストと交互配置） |
| `modality_flag` | 各トークン位置がテキスト/音声のどちらかを示すフラグ |
| `supervision_mask` | 損失計算の対象トークンを示すマスク |

テキストペア（方言テキスト → 標準語テキスト）は
[`dialect-to-standard-ja`](https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja)
の `dialect_standard_pairs.jsonl` と共通です。標準語音声はそのテキストを
edge-tts（`ja-JP-NanamiNeural`）で合成したものです。

## 元データ出典

- [`federerjiang/dialect.osaka`](https://huggingface.co/datasets/federerjiang/dialect.osaka)
- [`federerjiang/dialect.kumamoto`](https://huggingface.co/datasets/federerjiang/dialect.kumamoto)

標準語テキストは Qwen3-32B、標準語音声は edge-tts（`data/prepare_conversion_audio2audio.py`）で
自動生成しました。

## 関連リソース

- LoRA アダプタ: [`YujiYamaguchi/lfm25-audio-jp-dialect-audio2audio-lora`](https://huggingface.co/YujiYamaguchi/lfm25-audio-jp-dialect-audio2audio-lora)
- テキストのみ変換版データセット: [`YujiYamaguchi/dialect-to-standard-ja`](https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja)
- コード: [GitHub](https://github.com/YujiYamaguchi/hack-the-liquid-way-202606)
"""


def upload_model():
    repo_id = f"{HF_USER}/lfm25-audio-jp-dialect-audio2audio-lora"
    print(f"\n=== モデルリポジトリ作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    api.upload_file(
        path_or_fileobj=MODEL_README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  README.md アップロード完了")

    api.upload_file(
        path_or_fileobj="checkpoints/lora_audio2audio_full_v1/best/adapter_config.json",
        path_in_repo="adapter_config.json",
        repo_id=repo_id,
        repo_type="model",
    )
    print("  adapter_config.json アップロード完了")

    print("  adapter_model.safetensors アップロード中（約42MB）...")
    api.upload_file(
        path_or_fileobj="checkpoints/lora_audio2audio_full_v1/best/adapter_model.safetensors",
        path_in_repo="adapter_model.safetensors",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"  完了！ https://huggingface.co/{repo_id}")
    return repo_id


def upload_dataset():
    repo_id = f"{HF_USER}/dialect-to-standard-ja-speech"
    print(f"\n=== データセットリポジトリ作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    api.upload_file(
        path_or_fileobj=DATASET_README.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  README.md アップロード完了")

    for split in ("train", "val", "test"):
        folder = f"data/audio2audio_full_{split}"
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
