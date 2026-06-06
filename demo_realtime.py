"""
方言→標準語リアルタイム変換デモ

Usage:
    uv run python demo_realtime.py
    uv run python demo_realtime.py --lora_path checkpoints/lora_convert_v4/best
    uv run python demo_realtime.py --lora_path checkpoints/lora_convert_v4/best --wav input.wav --dialect_text "方言テキスト"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
SYSTEM_PROMPT = "次の方言音声を自然な標準語（です・ます調）に変換してください。"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(lora_path: str | None):
    lora_p = Path(lora_path) if lora_path else None

    # Full FT: adapter_config.json がなく model.safetensors がある場合
    if lora_p and lora_p.exists() and (lora_p / "model.safetensors").exists() and not (lora_p / "adapter_config.json").exists():
        print(f"Full FTモデルロード: {lora_path}")
        processor = LFM2AudioProcessor.from_pretrained(lora_p)
        model = LFM2AudioModel.from_pretrained(lora_p, device=DEVICE, dtype=torch.bfloat16)
        model.eval()
        print("準備完了")
        return model, processor

    print(f"モデルロード中: {MODEL_ID}")
    processor = LFM2AudioProcessor.from_pretrained(MODEL_ID)
    model = LFM2AudioModel.from_pretrained(MODEL_ID, device=DEVICE, dtype=torch.bfloat16)

    if lora_p and lora_p.exists() and (lora_p / "adapter_config.json").exists():
        print(f"LoRAアダプタロード: {lora_path}")
        from peft import LoraConfig, get_peft_model
        import safetensors.torch as st

        cfg = json.loads((lora_p / "adapter_config.json").read_text())
        lora_cfg = LoraConfig(
            r=cfg["r"], lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias=cfg.get("bias", "none"),
        )
        model = get_peft_model(model, lora_cfg)
        weights = st.load_file(str(lora_p / "adapter_model.safetensors"))
        model.load_state_dict(weights, strict=False)

    model.eval()
    print("準備完了")
    return model, processor


def record_audio_powershell(duration_s: int = 6) -> np.ndarray | None:
    """WSL2 環境で PowerShell 経由でマイク録音"""
    filename = f"dialect_rec_{uuid.uuid4().hex}.wav"
    win_temp = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "$env:TEMP"],
        text=True,
    ).strip()
    wsl_temp = subprocess.check_output(["wslpath", "-u", win_temp], text=True).strip()
    win_path = f"{win_temp}\\{filename}"
    wsl_path = f"{wsl_temp}/{filename}"

    ps_script = f"""Add-Type @"
using System.Runtime.InteropServices;
public class MCI {{
    [DllImport("winmm.dll")]
    public static extern int mciSendString(string s, System.Text.StringBuilder r, int l, System.IntPtr h);
    [DllImport("winmm.dll")]
    public static extern int waveInGetNumDevs();
}}
"@
if ([MCI]::waveInGetNumDevs() -eq 0) {{ Write-Error "マイクなし"; exit 1 }}
$p = '{win_path}'
[MCI]::mciSendString("open new Type waveaudio Alias rec", $null, 0, [System.IntPtr]::Zero) | Out-Null
[MCI]::mciSendString("set rec channels 1 bitspersample 16 samplespersec 16000 alignment 2 bytespersec 32000", $null, 0, [System.IntPtr]::Zero) | Out-Null
[MCI]::mciSendString("record rec", $null, 0, [System.IntPtr]::Zero) | Out-Null
Start-Sleep -Milliseconds {duration_s * 1000}
[MCI]::mciSendString("save rec $p", $null, 0, [System.IntPtr]::Zero) | Out-Null
[MCI]::mciSendString("close rec", $null, 0, [System.IntPtr]::Zero) | Out-Null
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(wsl_path):
        print(f"録音エラー: {result.stderr.strip()}")
        return None
    audio, _ = sf.read(wsl_path, dtype="float32")
    os.unlink(wsl_path)
    return audio


def apply_vad(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """silero-vad で発話区間だけ抽出"""
    try:
        from silero_vad import load_silero_vad, get_speech_timestamps
        vad_model = load_silero_vad()
        tensor = torch.from_numpy(audio).float()
        timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=sr, min_silence_duration_ms=500)
        if not timestamps:
            return audio
        start = timestamps[0]["start"]
        end = timestamps[-1]["end"]
        return audio[start:end]
    except Exception:
        return audio


def convert_dialect(
    model, processor, audio: np.ndarray, sr: int = 16000
) -> str:
    """方言音声 → 標準語テキスト (generate_sequential + LoRA)"""
    wav = torch.from_numpy(audio[None]).float().to(DEVICE)

    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text(SYSTEM_PROMPT)
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(wav, sr)
    chat.end_turn()
    chat.new_turn("assistant")

    text_tokens: list[int] = []
    with torch.no_grad():
        for token in model.generate_sequential(**chat, max_new_tokens=100, text_top_k=1):
            if token.numel() == 1:
                tid = token.item()
                if tid == 7:  # <|im_end|>
                    break
                text_tokens.append(tid)
            else:
                break

    if not text_tokens:
        return ""
    result = processor.text.decode(text_tokens, skip_special_tokens=True)
    return result if isinstance(result, str) else result[0]


_win_temp_cache: str | None = None

def _get_win_temp() -> tuple[str, str]:
    global _win_temp_cache
    if _win_temp_cache is None:
        _win_temp_cache = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "$env:TEMP"],
            text=True,
        ).strip()
    wsl_temp = subprocess.check_output(["wslpath", "-u", _win_temp_cache], text=True).strip()
    return _win_temp_cache, wsl_temp


def play_audio_powershell(audio: np.ndarray, sr: int = 24000) -> None:
    """WSL2 環境で PowerShell 経由で音声再生"""
    filename = f"play_{uuid.uuid4().hex}.wav"
    win_temp, wsl_temp = _get_win_temp()
    win_path = f"{win_temp}\\{filename}"
    wsl_path = f"{wsl_temp}/{filename}"

    # SoundPlayer はプロセス起動のたびに数百ms 失うため無音でパディング
    silence = np.zeros(int(1.0 * sr), dtype=np.float32)
    audio_padded = np.concatenate([silence, audio.astype(np.float32)])
    sf.write(wsl_path, audio_padded, sr)
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         f"(New-Object System.Media.SoundPlayer '{win_path}').PlaySync()"],
        check=False,
    )
    try:
        os.unlink(wsl_path)
    except OSError:
        pass


def prewarm_audio() -> None:
    """音声デバイスを初期化（最初の再生で先頭が切れる問題の回避）"""
    silence = np.zeros(int(0.4 * 16000), dtype=np.float32)
    play_audio_powershell(silence, 16000)


async def _tts_save(text: str, path: str, voice: str) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(path)


def synthesize_and_play_edgetts(text: str, voice: str = "ja-JP-NanamiNeural") -> None:
    """edge-tts で音声合成 → ffmpeg で WAV デコード → play_audio_powershell"""
    import tempfile
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(mp3_fd)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    try:
        asyncio.run(_tts_save(text, mp3_path, voice))
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path, "-ac", "1", "-ar", "24000", wav_path],
            capture_output=True, check=True,
        )
        audio, sr = sf.read(wav_path, dtype="float32")
        play_audio_powershell(audio, sr)
    finally:
        for p in (mp3_path, wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default="checkpoints/lora_convert_v4/best",
                        help="LoRAアダプタのパス")
    parser.add_argument("--duration", type=int, default=6, help="録音秒数")
    parser.add_argument("--wav", default=None, help="WAVファイルを直接入力（マイク不要）")
    parser.add_argument("--dialect_text", default=None, help="方言テキスト（画面表示用）")
    parser.add_argument("--voice", default="ja-JP-NanamiNeural", help="edge-tts 音声名")
    args = parser.parse_args()

    # lora_convert_v4 がなければ v2 にフォールバック
    lora_path = args.lora_path
    if not Path(lora_path).exists():
        fallback = "checkpoints/lora_convert_v2/best"
        if Path(fallback).exists():
            print(f"[注意] {lora_path} が見つかりません。{fallback} を使用します。")
            lora_path = fallback
        else:
            print("[注意] LoRAアダプタが見つかりません。ベースモデルで動作します。")
            lora_path = None

    model, processor = load_model(lora_path)

    print("\n" + "="*50)
    print("  方言→標準語変換デモ (LFM2.5-Audio-1.5B-JP)")
    print("="*50)

    if args.wav:
        # WAVファイル入力モード
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if args.dialect_text:
            print(f"  【方言】  {args.dialect_text}")
        print(f"  入力音声: {args.wav} ({len(audio)/sr:.1f}秒)")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 方言音声を再生（ピーク正規化で音量を標準語側に合わせる）
        print("[再生] 方言音声...")
        peak = np.abs(audio).max()
        audio_normalized = audio / (peak + 1e-8) * 0.9 if peak > 0 else audio
        play_audio_powershell(audio_normalized, sr)

        # LFM 推論
        print("[変換] LFM2.5-Audio-1.5B-JP + LoRA...")
        t0 = time.perf_counter()
        text = convert_dialect(model, processor, audio, sr)
        elapsed = time.perf_counter() - t0

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if args.dialect_text:
            print(f"  【方言】  {args.dialect_text}")
        print(f"  【標準語】 {text}")
        print(f"  推論時間: {elapsed:.1f}秒")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 標準語音声を再生（edge-tts）
        if text:
            print(f"[再生] 標準語音声（edge-tts: {args.voice}）...")
            synthesize_and_play_edgetts(text, voice=args.voice)

        return

    # マイク録音モード
    while True:
        try:
            input(f"\n[Enter]で録音開始（{args.duration}秒）、Ctrl+Cで終了...")
        except KeyboardInterrupt:
            print("\n終了します。")
            break

        print(f"録音中... ({args.duration}秒)")
        audio = record_audio_powershell(args.duration)
        if audio is None:
            print("録音失敗。マイクを確認してください。")
            continue

        print(f"録音完了 ({len(audio)/16000:.1f}秒) → VAD処理中...")
        audio = apply_vad(audio, 16000)
        print(f"発話区間: {len(audio)/16000:.1f}秒")

        if len(audio) < 3200:  # 0.2秒未満は無視
            print("音声が短すぎます。もう一度話してください。")
            continue

        print("変換中...")
        t0 = time.perf_counter()
        text = convert_dialect(model, processor, audio)
        elapsed = time.perf_counter() - t0

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  標準語: {text}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"推論時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
