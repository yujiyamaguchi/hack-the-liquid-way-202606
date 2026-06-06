# Hack the Liquid WAY — ハッカソン参加ガイド

**日程**: 2026年6月6日（土）〜 7日（日）  
**主催**: Liquid AI × WAY Equity Partners × AMD × Weights & Biases × HuggingFace  
**Discord**: Hack the Liquid WAY チャンネル  
**チーム規模**: 1〜3人

---

## スケジュール

### Day 1（6/6 土）

| 時間 | 内容 |
|------|------|
| 09:00 – 09:30 | 受付・ウェルカム |
| 09:30 – 10:15 | オープニング + テクニカルキックオフ + ユースケース探索 |
| 10:15 – 10:30 | ネットワーキング・チームビルディング |
| 10:30 – 12:30 | ハッキング開始 |
| 12:30 – 13:30 | ランチ |
| 13:30 – 16:30 | ハッキング |
| 16:30 – 17:00 | Day 1 まとめ |

### Day 2（6/7 日）

| 時間 | 内容 |
|------|------|
| 09:00 – 09:30 | ウェルカムバック |
| 09:30 – 13:30 | ハッキング + ランチ |
| **13:30** | **提出締切** |
| 14:00 – 16:00 | デモセッション（各チーム5分） |
| 16:00 – 16:30 | 審査員評価 + オーディエンス投票 |
| 16:30 – 17:00 | 表彰式・クロージング |
| 17:00 – 18:00 | ネットワーキング・ディナー |

---

## チャレンジテーマ

**Build industry-impact applications where LFMs are the critical unlock**

LFM（Liquid Foundation Models）が最善の解決策であり、かつ日本の産業において重大な課題を解決するアプリケーション/ワークフローを構築する。

LFMが他のモデルより優れる場面:

1. **プライバシー/データ主権** — データがデバイスやインフラ外に出せない場合
2. **レイテンシ** — ミリ秒単位の応答が求められる場合
3. **オフライン/省エネ** — クラウドラウンドトリップが物理的に不可能な場面（フィールドワーク、ウェアラブル、災害現場）
4. **ドメイン特化** — タスク特化LFM + 最適プロンプトが汎用大型モデルを上回る
5. **スケールコスト** — APIコールより計算サイクルが安い（常時稼働デバイス、高頻度処理）
6. **決定論的・制御可能な出力** — 再現性や規制要件がある場合

---

## トラック

### Track 1: LFM Application Track

公開またはfine-tuned LFMを使ったエンドユーザー向けソフトウェアアプリ/ワークフローを構築する。  
**AMD Ryzen AI PCでの動作デモが必須。**

### Track 2: LFM Audio/Fine-tuning Track

LFM（特にAudioモデル）をfine-tuneまたはadaptし、日本市場向けの実用的なユースケースでデモアプリを構築する。  
**HuggingFace Jobs ($150クレジット) でリモートGPUを使用。**

---

## 使用可能なモデル（LFMファミリー）

| モデル | 用途 |
|--------|------|
| LFM2.5 | テキスト全般 |
| LFM2.5-VL | ビジョン・言語 |
| LFM2.5-Audio | 音声 |
| Liquid Nanos | 軽量・エッジ向け |
| LFM-JP Collection | 日本語特化 |
| **New** LFM2.5-1.2B-JP-202606 | 最新日本語テキスト |
| **New** LFM2.5-Audio-1.5B-JP | 最新日本語音声 |
| **New** LFM2.5-VL-450M-Extract | 軽量ビジョン抽出 |
| **New** LFM2.5-VL-1.6B-Extract | 高精度ビジョン抽出 |

HuggingFace コレクション:
- [LFM2.5](https://huggingface.co/collections/liquid-ai/lfm25-67b2c7e6e1776e43d8eb11ff)
- [LFM2.5-VL](https://huggingface.co/collections/liquid-ai/lfm25-vl-67d7e8e3d0f8ac3ab5f91e36)
- [LFM2.5-Audio](https://huggingface.co/liquid-ai/LFM2.5-Audio-1.5B)
- [Liquid Nanos](https://huggingface.co/collections/liquid-ai/liquid-nanos-67e06e26a0af4ef99e9b1f4a)
- [LFM-JP Collection](https://huggingface.co/collections/liquid-ai/lfm-jp-collection)

---

## オンデバイスハードウェア（AMD Ryzen AI PC）

各チームに AMD Ryzen AI PC（XDNA 2 NPU搭載、最大50 TOPS）が貸与される。  
プリインストール済みランタイム:

| ランタイム | 対応モデル | 実行先 |
|-----------|-----------|--------|
| FastFlowLM | LFM2-1.2B（.q4nx） | NPUオフロード |
| llama.cpp + Vulkan | LFM2, LFM2-VL | 統合Radeon GPU |
| liquid-audio runtime | LFM2-Audio-1.5B | CPU / iGPU |
| LEAP SDK | デモアプリ | モバイル/デスクトップ |

フリート構成: Strix Halo / Strix Point / Krackan Point / Hawk Point / Lunar Lake（キックオフ時に担当SKUを発表）

---

## セットアップ（事前準備チェックリスト）

### 全員共通

- [ ] 政府発行の写真付きID（本人確認用）
- [ ] 自分のPC + 充電器
- [ ] Discord の「Hack the Liquid WAY」チャンネルに参加

### Track 2 向け追加セットアップ

- [ ] HuggingFace アカウント作成（ハッカソン登録時のメールで）
  - [https://huggingface.co/join](https://huggingface.co/join)
- [ ] Weights & Biases アカウント作成
  - [https://wandb.ai/site](https://wandb.ai/site)
- [ ] `uv` のインストール
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- [ ] プロジェクト初期化
  ```bash
  uv init my-hack-project
  cd my-hack-project
  uv add huggingface_hub wandb
  ```
- [ ] HuggingFace CLIにログイン
  ```bash
  uv run hf auth login
  ```
- [ ] Day1 に $150 HuggingFace Jobsクレジットを受け取り → claim

---

## HuggingFace Jobs クイックスタート（Track 2）

### ジョブの起動

```bash
uv run hf jobs run \
  --flavor a100-large \
  --timeout 3h \
  --secret HF_TOKEN \
  --secret WANDB_API_KEY \
  python:3.13 \
  bash -c "pip install transformers wandb && python train.py"
```

主要フラグ:
- `--flavor` — ハードウェアを選択（下記の価格表を参照）
- `--timeout` — 必ず設定すること。デフォルト30分。忘れたジョブが夜通し動くのを防ぐ
- `--secret` — HFトークン・W&Bの認証情報をJob環境に転送
- `--namespace` — チームのHF組織のクレジットプールに課金する場合に指定

### ハードウェア価格表（$150クレジットの使い方の目安）

| ワークロード | Flavor | $/時間 | $150で使える時間 |
|-------------|--------|--------|-----------------|
| LFM2-350M/1.2B SFT・LoRA | a100-large (1× A100 80GB) | $2.50 | ~60時間 |
| LFM2-1.2B フルfine-tune | l40sx1 (1× L40S 48GB) | $1.80 | ~83時間 |
| 合成データ生成・TTS推論 | l4x1 (1× L4 24GB) | $0.80 | ~187時間 |
| 軽量推論・評価スクリプト | t4-medium (16GB) | $0.60 | ~250時間 |
| LFM2-VL バッチ推論 | a10g-large (1× A10G 24GB) | $1.50 | ~100時間 |
| マルチGPU fine-tune（稀） | a100-large-x4 (4× A100 80GB) | $10.00 | ~15時間 |

### コスト管理

```bash
# 実行中ジョブの確認
uv run hf jobs list

# ジョブのキャンセル
uv run hf jobs cancel <job-id>

# 利用可能なハードウェアの一覧
uv run hf jobs hardware
```

- 課金は Job が Starting または Running の間のみ（ビルド中・失敗時は無料）
- 使用量は [huggingface.co/settings/billing](https://huggingface.co/settings/billing) の "Compute Usage" で確認

---

## 審査基準

| 基準 | 説明 |
|------|------|
| Fit to Challenge | 「なぜLFM？」「フロンティアLLMより優れている点は？」日本の産業への実際のインパクト |
| Creativity & Design | 実装のユニークさと思慮深さ |
| Quality & Completeness | ソフトウェア・プレゼンの完成度、最初の顧客獲得に向けた完全なピッチ |
| Resource Efficiency | 低コスト・低レイテンシ・低消費電力（エッジ/低コストデバイスへの展開でボーナス） |
| Track-Specific | トラック固有の評価 |

---

## 提出要件（Day 2 13:30 締切）

**各チームはトラックを1つ選択して提出。**

### 共通提出物

- [ ] **スライドデッキ（2〜4枚）**: 日本の問題/ユースケース、「なぜLFM？」、アプローチ、結果  
  ※ 日本語で発表する場合は英語スライド、英語で発表する場合は日本語スライドを用意
- [ ] **ライブデモ（5分）**: Day 2 デモセッションで実施
- [ ] **タグライン（1〜2行）** + **公開リポジトリのリンク**（登録時に同意したオープンソースプロジェクト規約に従う）
- [ ] **デモ資産フォルダ**（暗号化済み、パスワードは Discord の @liquid-yan に共有）
  - フォルダ名: `TEAMNAME_Track1_HackTheLiquidWAY_DemoAssets`（または Track2）
  - 内容: 60〜90秒のデモ動画、高解像度スクリーンショット、製品・チーム写真、キャプション/バイオ、README.txt（ファイル説明・デモセットアップ手順）
- [ ] **技術サマリー**（デッキまたはREADME.txtに記載）: 使用モデル・フレームワーク、計算環境、デバイス+レイテンシ/効率数値、アーキテクチャ図または主要技術革新

---

## Our Project

**Track 2: 方言音声リアルタイム標準語変換**

LFM2.5-Audio-1.5B-JP を LoRA fine-tuneし、方言音声（関西弁・熊本弁）を標準語音声に変換するリアルタイムデモを実現する。  
`generate_interleaved()` により、単一モデルで音声入力→標準語テキスト+音声を同時出力。

---

## モデル・データ

### LoRA アダプタ

ファインチューニング済みアダプタ（43MB）:  
**[YujiYamaguchi/lfm25-audio-jp-dialect-lora](https://huggingface.co/YujiYamaguchi/lfm25-audio-jp-dialect-lora)**

```bash
from huggingface_hub import snapshot_download
snapshot_download("YujiYamaguchi/lfm25-audio-jp-dialect-lora", local_dir="checkpoints/lora_convert_v4/best")
```

### 学習データ

加工済みデータセット（方言音声×標準語テキスト 2,600件）:  
**[YujiYamaguchi/dialect-to-standard-ja](https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja)**

元データ出典:
- [`federerjiang/dialect.osaka`](https://huggingface.co/datasets/federerjiang/dialect.osaka)
- [`federerjiang/dialect.kumamoto`](https://huggingface.co/datasets/federerjiang/dialect.kumamoto)

標準語テキストは Qwen3-32B で自動生成（`data/generate_standard_pairs.py` 参照）。

### ローカルでの学習再現

```bash
# 1. データ準備
uv run python data/prepare_conversion.py

# 2. LoRA 学習（RTX 5090 / A100 推奨）
uv run python train.py

# 3. デモ実行
bash demo_samples.sh
```

---

## リソース

- [Notion イベントガイド（原文）](https://liquidai.notion.site/Hack-the-Liquid-WAY-Event-Guide-370cbef042ad8120b019f78c480e41d8)
- [HuggingFace Jobs ドキュメント](https://huggingface.co/docs/huggingface_hub/guides/jobs)
- [HuggingFace Jobs 価格・ハードウェア一覧](https://huggingface.co/docs/hub/jobs-pricing)
- [LEAP SDK](https://leap.liquid.ai)
