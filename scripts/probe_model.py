"""
Probe LFM2.5-Audio-1.5B-JP to confirm:
- Model class (CausalLM or SpeechSeq2Seq)
- Processor type
- Linear layer names (for LoRA target_modules)
- Basic inference test
"""
import torch
from transformers import AutoConfig, AutoProcessor

MODEL_ID = "LiquidAI/LFM2.5-Audio-1.5B-JP"
FALLBACK_ID = "LiquidAI/LFM2.5-Audio-1.5B"

def probe(model_id: str):
    print(f"\n=== Probing {model_id} ===")

    # Config
    print("\n-- Config --")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"model_type: {cfg.model_type}")
    print(f"architectures: {cfg.architectures}")
    print(f"config keys: {list(cfg.to_dict().keys())[:15]}")

    # Processor
    print("\n-- Processor --")
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"processor type: {type(proc).__name__}")
    print(f"processor attrs: {[a for a in dir(proc) if not a.startswith('_')][:10]}")

    # Model load (bfloat16, CPU first to check structure)
    print("\n-- Model load (cpu, bfloat16) --")

    # Try CausalLM first
    loaded_class = None
    for cls_name in ["AutoModelForCausalLM", "AutoModelForSpeechSeq2Seq"]:
        try:
            import importlib
            transformers = importlib.import_module("transformers")
            cls = getattr(transformers, cls_name)
            model = cls.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            loaded_class = cls_name
            print(f"Loaded with: {cls_name}")
            break
        except Exception as e:
            print(f"{cls_name} failed: {e}")

    if loaded_class is None:
        print("ERROR: Could not load model")
        return

    # Linear layers
    print("\n-- Linear layers (LoRA targets) --")
    linear_layers = [
        (name, mod.in_features, mod.out_features)
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    ]
    print(f"Total linear layers: {len(linear_layers)}")
    print("First 30:")
    for name, in_f, out_f in linear_layers[:30]:
        print(f"  {name}  ({in_f} -> {out_f})")

    # Unique suffixes (useful for LoRA target_modules)
    suffixes = sorted(set(name.split(".")[-1] for name, _, _ in linear_layers))
    print(f"\nUnique layer name suffixes: {suffixes}")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total/1e9:.2f}B")

    # Try a dummy forward pass with audio
    print("\n-- Dummy inference test --")
    try:
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        inputs = proc(dummy_audio, sampling_rate=16000, return_tensors="pt")
        print(f"Processor output keys: {list(inputs.keys())}")
        print(f"Input shapes: { {k: v.shape for k, v in inputs.items()} }")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        decoded = proc.decode(out[0], skip_special_tokens=True)
        print(f"Generated (silence): '{decoded}'")
        print("Inference: OK")
    except Exception as e:
        print(f"Inference failed: {e}")

    print(f"\n=== RESULT: Use {loaded_class} with {model_id} ===")
    return loaded_class, suffixes


if __name__ == "__main__":
    result = probe(MODEL_ID)
    if result is None:
        print(f"\nFalling back to {FALLBACK_ID}")
        probe(FALLBACK_ID)
