# CLAUDE.md — Hack the Liquid WAY ハッカソン

## プロジェクト概要

Liquid AI主催ハッカソン（2026/06/06-07）の開発作業ディレクトリ。  
テーマ: **LFM（Liquid Foundation Models）が必然的な解決策となる、産業インパクトのあるアプリケーションを構築する**

提出締切: **2026/06/07 13:30**

## このプロジェクトの内容

**Track 2: 構音障害・アクセント音声向けASRの個人化**

LFM2.5-Audio-1.5B-JP（ハッカソンで新たに提供）を LoRA/Adapter でfine-tuneし、構音障害（脳性麻痺・ALS）話者向けのパーソナライズドASRを実現する。

- **使用モデル**: `liquid-ai/LFM2.5-Audio-1.5B-JP`（優先）、フォールバック: `liquid-ai/LFM2.5-Audio-1.5B`
- **ベースライン**: `nvidia/canary-180m-flash`（FastConformer 115M）
- **データセット**: TORGO（CP/ALS話者、WAV 16kHz）、ALS公開データ
- **手法**: LoRA / Adapter / Full Fine-tuning の比較。少量データ（5%〜50%）で評価
- **実験管理**: Weights & Biases（W&B）
- **デモ環境**: AMD Ryzen AI PC（liquid-audio runtime）
- **詳細**: [proposal.md](proposal.md)

---

## トラック

- **Track 1**: 既存またはfine-tuned LFMを使ったエンドユーザー向けアプリ（AMD Ryzen AI PC動作デモ必須）
- **Track 2**: LFM（特にAudioモデル）のfine-tune/adapt + デモアプリ（HuggingFace Jobs $150クレジット使用）

## 技術スタック方針

### パッケージ管理

`pip` 直接実行ではなく `uv` を使う。

```bash
uv add <package>          # 依存追加
uv run <command>          # 仮想環境内で実行
uv run python script.py   # スクリプト実行
```

### 使用モデル

HuggingFace の `liquid-ai` オーガニゼーション配下のモデルを使用:

- `liquid-ai/LFM2.5-1.2B` — テキスト全般
- `liquid-ai/LFM2.5-1.2B-JP-202606` — 日本語特化（最新）
- `liquid-ai/LFM2.5-Audio-1.5B-JP` — 日本語音声（最新）
- `liquid-ai/LFM2.5-VL-1.6B-Extract` — ビジョン抽出（最新）

## 評価で重視される観点

審査基準の優先順位を意識してコードを書く:

1. **「なぜLFM？」** — クラウドLLM（GPT-4等）ではなくLFMを使う必然性を明確に
2. **日本の産業への実インパクト** — 実際の問題を解決していること
3. **Resource Efficiency** — 低レイテンシ・低コスト・省エネ。on-device/エッジ展開でボーナス点
4. **完成度** — ライブデモで実際に動くことが前提

## 提出物の構成

```
TEAMNAME_Track1_HackTheLiquidWAY_DemoAssets/
├── README.txt              # ファイル説明・デモセットアップ手順
├── demo_video.mp4          # 60〜90秒のデモ動画
├── screenshots/            # 高解像度スクリーンショット
└── photos/                 # 製品・チーム写真
```

スライドデッキ（2〜4枚）に必ず含める:
1. 日本の問題/ユースケース
2. 「なぜLFM？」（クラウドLLMとの差別化）
3. アプローチ・アーキテクチャ
4. 結果・デモ

## HuggingFace Jobs（Track 2のみ）

```bash
uv run hf jobs run \
  --flavor a100-large \
  --timeout 3h \
  --secret HF_TOKEN \
  --secret WANDB_API_KEY \
  python:3.13 \
  bash -c "pip install transformers wandb && python train.py"
```

**コスト注意**: `--timeout` を必ず設定。`uv run hf jobs cancel <job-id>` で不要なジョブを停止。

## AMD Ryzen AI PC（デモ環境）

デモ当日に割り当てられるPC上のランタイム:
- `FastFlowLM` — .q4nx形式でNPUオフロード（LFM2-1.2Bテキスト）
- `llama.cpp + Vulkan` — LFM2/LFM2-VL推論（統合Radeon GPU）
- `liquid-audio` — LFM2-Audio-1.5B（CPU/iGPU）
- `LEAP SDK` — モバイル/デスクトップデモ

ローカル開発時から推論負荷・レイテンシを意識した実装にする。
