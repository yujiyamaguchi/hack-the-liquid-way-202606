================================================================
  Hack the Liquid WAY 2026 — Track 2 Submission
  方言音声リアルタイム標準語変換デモ
  Dialect-to-Standard Japanese Real-time Voice Conversion
================================================================

チーム名: チーム山口
提出日: 2026/06/07

----------------------------------------------------------------
ファイル構成
----------------------------------------------------------------

README.txt          本ファイル
demo_video.mp4      デモ動画 (60〜90秒)
slides_en.html      プレゼンスライド（英語版・本番プレゼン用、ブラウザで開いて使用）
slides_ja.html      プレゼンスライド（日本語版・内容確認用）
screenshots/        スクリーンショット
photos/             チーム・環境写真

----------------------------------------------------------------
公開モデル・データセット (HuggingFace Hub)
----------------------------------------------------------------

【方言音声 → 標準語「音声＋テキスト」直接生成（最新・デモで使用）】
  LoRA  : https://huggingface.co/YujiYamaguchi/lfm25-audio-jp-dialect-audio2audio-lora
  データ: https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja-speech

【方言音声 → 標準語テキスト変換（旧版）】
  LoRA  : https://huggingface.co/YujiYamaguchi/lfm25-audio-jp-dialect-lora
  データ: https://huggingface.co/datasets/YujiYamaguchi/dialect-to-standard-ja

----------------------------------------------------------------
デモ概要
----------------------------------------------------------------

関西弁・熊本弁の音声を入力すると、LFM2.5-Audio-1.5B-JP が
LoRA アダプタ経由で標準語の「テキストと音声」を直接・同時に生成します
（外部TTSエンジン不使用、モデル自身の声で出力）。

  方言音声入力 → [LFM2.5-Audio-1.5B-JP + LoRA (43MB)] → 標準語テキスト＋音声を同時生成（generate_interleaved）

ASR + LLM + TTS の3段パイプラインを介さず、単一の1.5Bモデル・単一推論で
方言音声 → 標準語音声の End-to-End 変換を実現しています。
（旧版: 方言音声 → 標準語テキスト変換 + edge-tts読み上げ。比較用に同梱、後述）

----------------------------------------------------------------
セットアップ手順
----------------------------------------------------------------

前提環境:
  - CUDA対応GPU (8GB VRAM以上)
  - Python 3.12+
  - uv (https://docs.astral.sh/uv/)
  - WSL2 または Linux (音声I/OにWindowsマイク/スピーカー使用)

1. 依存パッケージのインストール:
   uv sync

2. モデルのダウンロード（初回のみ、自動）:
   HuggingFace より LiquidAI/LFM2.5-Audio-1.5B-JP を自動取得

3. デモ実行（音声→音声 直接生成、メインデモ）:
   uv run python demo_audio2audio.py
   → テストデータから代表3サンプルを自動選択し、各サンプルについて
     方言音声(原音)の再生 → モデルによる標準語「テキスト＋音声」の
     直接生成・再生、を順番に実行（外部TTS不使用・モデル自身の声）

4. （比較用・旧版）方言音声 → 標準語テキスト変換 + edge-tts読み上げ:
   uv run python demo_realtime.py \
     --lora_path checkpoints/lora_convert_v4/best \
     --wav path/to/dialect.wav \
     --dialect_text "方言テキスト（画面表示用）"

   サンプル5件を順番に実行: bash demo_samples.sh

5. マイク録音モード（リアルタイム・旧版のみ対応）:
   uv run python demo_realtime.py --lora_path checkpoints/lora_convert_v4/best
   → Enterキーで6秒録音 → 標準語テキスト表示 + 音声読み上げ

----------------------------------------------------------------
学習済みチェックポイント
----------------------------------------------------------------

checkpoints/lora_audio2audio_full_v1/best  LoRA 音声→音声直接生成 (val_loss 1.6966→1.3572, CER 0.362) ← メインデモ採用
checkpoints/lora_convert_v4/best   LoRA 音声→テキスト変換 (val_loss=0.934, CER 0.40) ← 比較用デモ採用
checkpoints/lora_convert_v5/best   LoRA (val_loss=0.961) ← さらに学習したが悪化
checkpoints/full_ft_v1/best        Full Fine-tune (val_loss=0.908) ← 不採用

----------------------------------------------------------------
ファインチューニング詳細
----------------------------------------------------------------

ベースモデル : LiquidAI/LFM2.5-Audio-1.5B-JP
手法        : LoRA (r=16, alpha=32, dropout=0.05)
               対象層: q_proj/k_proj/v_proj/out_proj, w1/w2/w3 (FFN)
               訓練パラメータ: 11M / 1,464M = 0.76%  アダプタ: 43MB
学習環境    : RTX 5090
               ※ ハッカソン提供クレジット: HuggingFace Jobs (A100) も利用可
デモ環境    : エッジデバイス（AMD Ryzen AI PC 上の liquid-audio ランタイムで動作可能）
学習ステップ: 1000 steps（lora_convert_v4 / lora_audio2audio_full_v1 共通）
ハイパーパラメータ:
               LR=1e-4 (warmup 50steps → cosine decay)
               batch_size=4, context_length=256
               ※ 公式レシピ準拠
               https://github.com/Liquid4All/liquid-ai-way-amd-huggingface-wandb-tokyo-hackathon-2026/blob/main/examples/audio/audio_finetune_walkthrough.ipynb

学習データ  :
  出典: federerjiang/dialect.osaka + federerjiang/dialect.kumamoto (HuggingFace)
  加工: 方言テキスト → Qwen3-32B で標準語テキストを自動生成
  形式: 方言音声 → 標準語テキスト（2,600件）
  分割: train 80% / val 10% / test 10%（seed=42）

  ※ メインデモ採用の lora_audio2audio_full_v1（音声→音声直接生成）は、
  上記と全く同一のデータ・分割・レシピ（MAX_STEPS=1000, LR=1e-4）で、
  標準語テキストを edge-tts で音声合成し、各ペアを InterleavedSegment
  （テキストと音声フレームを交互配置する専用データ型）として
  generate_interleaved の出力パターンと一致する形式で学習データを構築
  （これが「テキストと音声を交互に生成する」こと自体を学習させる鍵となる工夫）。
  音声出力という難易度の高いタスクでも同条件で同等以上の精度・収束が
  得られることを検証するコントロール実験として設計。

----------------------------------------------------------------
技術的なポイント（なぜ LFM か）
----------------------------------------------------------------

LFM2.5-Audio-1.5B-JP はテキストトークンと音声フレームを
同一モデル・単一推論で生成できるアーキテクチャを持ちます。
今回、これを活かして「方言音声 → 標準語テキスト＋音声」を
1モデル・1推論で直接生成する End-to-End 変換を実際に学習・実証しました
（lora_audio2audio_full_v1、メインデモ）。
ASR + テキスト変換 + TTS の3段パイプラインをクラウドに頼ることなく、
1モデルで代替できることを示しています。

クラウドLLMとの差別化:
  - 音声Fine-tuneが可能（LoRA 43MB のみ）
  - エッジデバイス上の liquid-audio ランタイムで動作
  - 完全ローカル処理でプライバシー保護
  - ネットワーク遅延なし

----------------------------------------------------------------
ファインチューニング知見・ノウハウ
----------------------------------------------------------------

【公式レシピとの対応】
公式ノートブック（上記URL）は TTS（テキスト→音声）ファインチューニングのレシピ。
我々のタスク（方言音声→標準語テキスト）はその逆方向であり、データ形式が異なる。

  公式レシピ          : system(TTS指示) + user(テキスト) → assistant(音声)
  lora_convert_v4     : system(変換指示) + user(方言音声) → assistant(標準語テキスト)
  lora_audio2audio_full_v1: system(変換指示) + user(方言音声)
                        → assistant(標準語テキスト＋音声を InterleavedSegment で同時出力)

LR・バッチサイズ・context_length・warmup等のハイパーパラメータは公式レシピに準拠。

【<|text_end|> トークン】
アシスタント出力末尾に <|text_end|> を付与することで、
モデルが「テキスト生成の終端」と「音声フレームへの切り替えタイミング」を学習する。
このトークンがないとテキスト生成が適切に終了せず、音声生成への移行ができない。

【アシスタントターンのみ損失計算】
liquid_audio の LFM2AudioChatMapper が自動的に supervision_mask を生成し、
system/user ターンのトークンは損失計算から除外される（明示的な実装不要）。

【LoRA vs Full Fine-tuning】
Full FT は val_loss でやや優位（0.908 vs 0.934）だが今回は LoRA を採用。
理由: Full FT は全重みを更新するため depthformer（音声デトークナイザー）も変更される可能性があり、
     音声出力への影響が未検証。LoRA は対象層を限定するため音声デコード能力を保持しやすい。

【val_loss の解釈】
val_loss はテキストヘッド + 音声8コードブックヘッドの複合損失であり、
言語モデルの perplexity とは直接比較できない。絶対値より改善の軌跡で評価する。

【generate_sequential vs generate_interleaved】
generate_sequential: テキストトークン → 音声フレームを逐次生成。テキスト出力が安定。
  → lora_convert_v4（音声→テキスト変換、比較用デモ）で採用。
generate_interleaved: テキストと音声を交互に生成し、End-to-End 音声→音声変換が可能。
  → lora_audio2audio_full_v1（メインデモ）で実証。学習データを
    InterleavedSegment 型として generate_interleaved の出力パターンと
    一致させることで、安定した「テキスト＋音声」同時生成を学習できることを確認。
    （当初は今後の課題としていたが、本ハッカソン中に実現・検証済み）

【greedy decoding（top_k=1）の採用】
変換タスクは「正解がほぼ一つ」のタスクのため、
毎回最も確率の高いトークンを選択する greedy decoding を使用。
再現性のある安定した出力が得られ、デモ品質の均一化に有効。
