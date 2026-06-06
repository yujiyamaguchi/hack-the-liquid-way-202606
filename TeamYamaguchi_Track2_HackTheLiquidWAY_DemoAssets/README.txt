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
slides.html         プレゼンスライド（ブラウザで開いて使用）
screenshots/        スクリーンショット
photos/             チーム・環境写真

----------------------------------------------------------------
デモ概要
----------------------------------------------------------------

関西弁・熊本弁の音声を入力すると、LFM2.5-Audio-1.5B-JP が
LoRA アダプタ経由で標準語テキストをリアルタイム変換します。

  方言音声入力 → [LFM2.5-Audio-1.5B-JP + LoRA (43MB)] → 標準語テキスト表示 + edge-tts 読み上げ

外部音声APIを使わず、単一の1.5Bモデルが音声→テキスト変換を担います。
本来の目標は End-to-End 音声→音声変換（generate_interleaved）でしたが、
今回は安定性確保のためテキスト出力に絞り、音声読み上げに edge-tts を使用しています。

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

3. WAVファイルでデモ実行:
   uv run python demo_realtime.py \
     --lora_path checkpoints/lora_convert_v4/best \
     --wav path/to/dialect.wav \
     --dialect_text "方言テキスト（画面表示用）"

4. サンプル5件を順番に実行:
   bash demo_samples.sh          # 全件
   bash demo_samples.sh 3        # 3番目のみ

5. マイク録音モード（リアルタイム）:
   uv run python demo_realtime.py --lora_path checkpoints/lora_convert_v4/best
   → Enterキーで6秒録音 → 標準語テキスト表示 + 音声読み上げ

----------------------------------------------------------------
学習済みチェックポイント
----------------------------------------------------------------

checkpoints/lora_convert_v4/best   LoRA (val_loss=0.934) ← デモ採用
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
学習ステップ: 1000 steps (lora_convert_v4)
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

----------------------------------------------------------------
技術的なポイント（なぜ LFM か）
----------------------------------------------------------------

LFM2.5-Audio-1.5B-JP はテキストトークンと音声フレームを
同一モデル・単一推論で生成できるアーキテクチャを持ちます。
これにより、ASR + テキスト変換 + TTS の3段パイプラインを
1モデルで代替できる可能性があります（今後の課題として取り組み中）。

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

  公式レシピ  : system(TTS指示) + user(テキスト) → assistant(音声)
  本プロジェクト: system(変換指示) + user(方言音声) → assistant(標準語テキスト)

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
generate_interleaved: テキストと音声を交互に生成（End-to-End 音声変換が可能）。
  → 今回は generate_interleaved による音声出力の品質が安定せず、
    テキスト出力のみ（generate_sequential + greedy decoding / top_k=1）を採用。
    音声 End-to-End 変換は今後の課題。

【greedy decoding（top_k=1）の採用】
変換タスクは「正解がほぼ一つ」のタスクのため、
毎回最も確率の高いトークンを選択する greedy decoding を使用。
再現性のある安定した出力が得られ、デモ品質の均一化に有効。
