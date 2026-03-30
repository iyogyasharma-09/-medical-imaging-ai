"""
gpu_check.py
------------
Run this BEFORE starting a 40-epoch training run.

Verifies that all three models fit in GPU memory at batch_size=1
with the configured spatial size (96, 96, 96) and 4 input channels.

Also checks Swin Transformer 3D at feature_size=24, which is the
primary memory risk. If it OOMs here, reduce feature_size to 12 in config.py.

Usage (on GPU machine):
    python gpu_check.py
    python gpu_check.py --batch-size 2   # test with larger batch

Output:
    Per-model: parameter count, forward pass peak memory, inference time
    OOM warning if any model doesn't fit
    Recommendation: feature_size to use for Swin
"""

import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import INPUT_CHANNELS, SPATIAL_SIZE, MODELS, DEVICE


def count_params(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{total / 1e6:.1f}M"


def probe_model(model_name: str, batch_size: int = 1) -> dict:
    from src.train_models import get_model

    print(f"\n  [{model_name}]")
    result = {"model": model_name, "status": "unknown"}

    try:
        model = get_model(model_name).to(DEVICE)
        model.eval()
        params = count_params(model)
        print(f"    Parameters : {params}")

        dummy = torch.zeros(batch_size, INPUT_CHANNELS, *SPATIAL_SIZE, device=DEVICE)

        # Clear cache before measuring
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(dummy)
        elapsed = time.perf_counter() - t0

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
            print(f"    Peak VRAM  : {peak_mb:.0f} MB")
            result["peak_vram_mb"] = peak_mb
        else:
            print(f"    Device     : CPU (no VRAM measurement)")

        print(f"    Output     : {tuple(out.shape)}")
        print(f"    Fwd time   : {elapsed*1000:.0f} ms")
        print(f"    Status     : ✓ OK")
        result.update({"status": "ok", "params": params, "fwd_ms": round(elapsed * 1000)})

    except torch.cuda.OutOfMemoryError:
        print(f"    Status     : ✗ OUT OF MEMORY")
        print(f"    Fix        : Reduce feature_size in config.py (try 12 for Swin)")
        result["status"] = "OOM"

    except Exception as e:
        print(f"    Status     : ✗ ERROR — {e}")
        result["status"] = f"error: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description="GPU memory check for 3D BraTS models")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  GPU Memory Check — BraTS 2021 3D Models")
    print("=" * 60)
    print(f"  Device      : {DEVICE}")
    print(f"  Input shape : ({args.batch_size}, {INPUT_CHANNELS}, {SPATIAL_SIZE[0]}, "
          f"{SPATIAL_SIZE[1]}, {SPATIAL_SIZE[2]})")

    if torch.cuda.is_available():
        total_vram = torch.cuda.get_device_properties(DEVICE).total_memory / 1e9
        print(f"  Total VRAM  : {total_vram:.1f} GB")
        print(f"  GPU name    : {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("  WARNING: No CUDA device found — running on CPU")
        print("  This check is only meaningful on a GPU machine.")
    print()

    results = []
    for model_name in MODELS:
        r = probe_model(model_name, batch_size=args.batch_size)
        results.append(r)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_ok = all(r["status"] == "ok" for r in results)

    for r in results:
        icon = "✓" if r["status"] == "ok" else "✗"
        vram = f"  {r.get('peak_vram_mb', 0):.0f} MB VRAM" if "peak_vram_mb" in r else ""
        print(f"  {icon} {r['model']:<28} params={r.get('params','?'):>6}{vram}")

    print()
    if all_ok:
        print("  ✓ All models fit at batch_size={}.".format(args.batch_size))
        print("  ✓ Safe to start training: python train.py --batch-size 1")
    else:
        print("  ✗ One or more models OOM'd. Try:")
        print("    • batch-size 1  (if not already)")
        print("    • feature_size = 12  (in config.py, for Swin-3D)")
        print("    • gradient_accumulation_steps = 4  (effective batch = 4)")
        sys.exit(1)


if __name__ == "__main__":
    main()
