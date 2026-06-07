---
marp: true
theme: default
paginate: true
style: |
  section { font-size: 22px; }
  h1 { font-size: 32px; }
  h2 { font-size: 28px; color: #1a1a2e; }
  table { font-size: 18px; }
  blockquote { font-size: 20px; background: #f0f4ff; border-left: 4px solid #4a90e2; padding: 8px 16px; }
  code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
---

# Dialect Speakers Are Locked Out of Voice AI

**Hack the Liquid WAY 2026 — Track 2 / Team Yamaguchi**

---

## Slide 1: Problem / Use Case

- Japan has many regional dialects (Kansai-ben, Kyushu-ben, Tohoku-ben, etc.)
- Existing ASR / voice AI assumes Standard Japanese → **dialect speakers get rejected**
- Especially serious for elderly and rural residents: "I can't operate things by voice"
- Examples: smart speakers, voice input, car navigation, call-center AI

> When I speak in Kansai dialect, my smart speaker just doesn't understand me.
> Having to "translate" myself into Standard Japanese while speaking is exhausting.

---

## Slide 2: Why LFM?

**LFM2.5-Audio-1.5B-JP turns "dialect speech → standard speech" into a single model, single inference pass**

| | Cloud LLM (typical API pipeline) | LFM2.5-Audio-1.5B-JP |
|---|---|---|
| Speech-to-speech | 3-stage: ASR + LLM + TTS | **1 model, 1 inference (text & audio generated jointly)** |
| Edge deployment | Not possible (requires cloud) | **Runs on edge devices** |
| Latency | High (network round-trips, 3 stages) | **Low (on-device, single pass)** |
| Privacy | Voice data leaves to the cloud | **Fully local processing** |
| Customization | Cannot fine-tune the voice output | **A 43MB LoRA adapts the whole speech-to-speech mapping** |

→ We actually trained and demonstrated an end-to-end **"dialect speech → standard speech"**
LoRA (see Slide 4). This is only possible because LFM **generates text tokens and audio
frames from the same model** — a pipeline of cloud ASR + LLM + TTS simply cannot deliver
this kind of unified, on-device conversion.

---

## Slide 3: Approach / Architecture

**The pipeline we built (speech-to-speech, end-to-end, single model)**:
```
[Dialect speech, 16kHz] → [LFM2.5-Audio-1.5B-JP + LoRA (43MB)] → [Standard-Japanese text + speech, generated jointly]
```

**Training data**:
- Source: `federerjiang/dialect.osaka` + `federerjiang/dialect.kumamoto` (dialect speech + transcripts); we used Qwen3-32B to auto-generate the corresponding Standard-Japanese text
- We synthesized that standard text into speech with edge-tts and packaged each pair as an `InterleavedSegment` (a data type that interleaves text tokens and audio frames), matching the output pattern of `generate_interleaved` — the key trick behind learning to generate text and audio in alternation
- Training pairs: dialect speech → standard text + speech (Osaka-ben + Kumamoto-ben,
  2,600 samples total; train/val/test = 2,077 / 260 / 258)

**LoRA config**: r=16, alpha=32 / target modules: q/k/v/out_proj, w1/w2/w3 / 0.76% trainable
params / trained on an RTX 5090 / MAX_STEPS=1000, LR=1e-4

**Controlled experiment**: we trained this audio-output model on the **exact same dataset
and recipe** as our existing "dialect speech → standard text" model (lora_convert_v4), to
verify that adding end-to-end speech generation doesn't hurt convergence or accuracy.

---

## Slide 4: Results

**Training convergence** (same data & recipe; absolute val_loss values aren't directly
comparable because the loss objectives differ — text-only vs. text+audio):
- lora_convert_v4 (text-only output): converges to val_loss 0.934
- lora_audio2audio_full_v1 (text + audio, jointly generated): val_loss decreases
  monotonically from **1.6966 → 1.3572**, still improving at step 1000

**Accuracy (text-component CER — the one metric directly comparable across both models)**:

| Model | Output | Text-component CER |
|-------|--------|---------------------|
| lora_convert_v4 (existing, adopted) | text only | 0.40 |
| lora_audio2audio_full_v1 (new) | **text + speech, jointly generated** | **0.362** |

→ Generating speech alongside text does not degrade accuracy — if anything, it's slightly better.

**Independently verifying the generated speech** (using a third-party ASR, `kotoba-whisper-v2.1`, that played no role in training):
- Clean reference audio: CER **4.2%** (confirms the judge is reliable)
- Generated speech: CER **37.7%** (same level as the 36.2% text-component CER → genuinely intelligible, coherent Standard Japanese, not garbled noise)

---

## Slide 5: Demo

**[Demo video] Speak in dialect — the model directly generates standard-Japanese speech
in its own voice**

**Conversion examples (from the demo, unseen test samples)**:

| Dialect (spoken input) | Standard Japanese (speech + text generated directly by the model) |
|------------------------|---------------------------------------------------------------------|
| 川向こうんジョイフル本田さん行った帰りん撮った写真**たい** (Kumamoto-ben) | 川向こうのジョイフルホンダさんに行った帰りに撮った写真**です**。 |
| 食費はかさむ一方**たい** (Kumamoto-ben) | 食費はかさむ一方**です**。 |
| **ボクが買ったんは**大学**入って**からやねんな (Osaka-ben) | **私が買ったのは**大学**に入って**からです。 |

**Next steps**: improve voice quality / naturalness / prosody, expand dialect coverage
(Tohoku-ben, Hakata-ben, etc.), personalize to individual speaking styles, and ship a
fully on-device implementation on the edge runtime.
