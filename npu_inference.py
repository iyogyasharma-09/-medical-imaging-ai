"""
npu_inference.py
-----------------
Run BraTS 2023 inference on Intel AI Boost NPU (or Intel iGPU)
using OpenVINO after training on GPU.

DEVICE PRIORITY
===============
  1. NPU    — Intel AI Boost (fastest low-power, fp16)
  2. GPU.0  — Intel Arc iGPU (fallback)
  3. GPU.1  — NVIDIA RTX A1000 via OpenVINO
  4. CPU    — last resort

USAGE
=====
  # Run inference on NPU using a trained model
  python npu_inference.py --model resnet18_medicalnet --task GLI

  # Use specific OpenVINO device
  python npu_inference.py --model resnet18_medicalnet --ov-device GPU.0

  # Benchmark all devices
  python npu_inference.py --model resnet18_medicalnet --benchmark
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_TEST_MANIFEST,
    INPUT_CHANNELS,
    MODELS_DIR,
    RESULTS_DIR,
    SPATIAL_SIZE,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms


# ===========================================================================
# OpenVINO device selection
# ===========================================================================

DEVICE_PRIORITY = ["NPU", "GPU.0", "GPU.1", "CPU"]

def get_best_ov_device(prefer: Optional[str] = None) -> str:
    """Returns the best available OpenVINO device."""
    import openvino as ov
    core    = ov.Core()
    avail   = core.available_devices
    print(f"[OpenVINO] Available devices: {avail}")

    if prefer and prefer in avail:
        return prefer
    for d in DEVICE_PRIORITY:
        if d in avail:
            print(f"[OpenVINO] Selected device: {d}")
            return d
    return "CPU"


# ===========================================================================
# Load OpenVINO compiled model
# ===========================================================================

def load_ov_model(model_name: str, ov_device: str):
    """
    Load the OpenVINO IR model and compile it for the chosen device.
    Falls back to re-exporting from PyTorch checkpoint if IR not found.
    """
    import openvino as ov

    xml_path = MODELS_DIR / f"{model_name}_ov" / "model.xml"

    if not xml_path.exists():
        print(f"[NPU] OpenVINO IR not found at {xml_path}")
        print(f"[NPU] Run training first: python train_brats2023.py --model {model_name}")
        print(f"[NPU] Or export manually: python train_brats2023.py (auto-exports after training)")
        sys.exit(1)

    core     = ov.Core()
    ov_model = core.read_model(str(xml_path))

    # Device-specific performance hints
    config = {}
    if ov_device == "NPU":
        config["PERFORMANCE_HINT"] = "THROUGHPUT"
    elif ov_device.startswith("GPU"):
        config["PERFORMANCE_HINT"] = "LATENCY"

    compiled = core.compile_model(ov_model, ov_device, config)
    print(f"[OpenVINO] Model compiled for {ov_device}: {xml_path.parent.name}")
    return compiled


# ===========================================================================
# NPU inference loop
# ===========================================================================

def run_npu_inference(
    model_name: str,
    task:       str = "GLI",
    label_col:  str = "label_grade",
    ov_device:  Optional[str] = None,
    batch_size: int = 1,
) -> Dict:
    """
    Run inference on the test set using OpenVINO on NPU/iGPU.

    Returns a results dict with per-subject predictions and accuracy.
    """
    import openvino as ov

    if ov_device is None:
        ov_device = get_best_ov_device()

    compiled = load_ov_model(model_name, ov_device)
    infer_req = compiled.create_infer_request()

    # ---- Load test manifest ----
    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col=label_col)
    if task != "ALL":
        df_test = df_test[df_test["task"] == task].reset_index(drop=True)

    print(f"[NPU] Inference: {len(df_test)} subjects  |  device={ov_device}")

    test_ds = BraTS2023Dataset(df_test, transforms=get_val_transforms(), label_col=label_col)

    all_preds  = []
    all_labels = []
    all_probs  = []
    times      = []

    for i in range(len(test_ds)):
        sample = test_ds[i]
        img    = sample["image"].unsqueeze(0).numpy()   # (1, 4, D, H, W)
        label  = int(sample["label"].item())

        t0 = time.perf_counter()
        infer_req.infer({"image": img})
        logits = infer_req.get_output_tensor(0).data   # (1, num_classes)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms

        probs = _softmax(logits[0])
        pred  = int(np.argmax(probs))

        all_preds.append(pred)
        all_labels.append(label)
        all_probs.append(probs.tolist())

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(test_ds)}] pred={pred} true={label} "
                  f"prob={probs.max():.3f}  latency={elapsed*1000:.1f}ms")

    # ---- Metrics ----
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    results = {
        "model_name":    model_name,
        "ov_device":     ov_device,
        "task":          task,
        "label_col":     label_col,
        "n_subjects":    len(df_test),
        "accuracy":      float(acc),
        "f1_weighted":   float(f1),
        "mean_latency_ms": float(np.mean(times)),
        "p95_latency_ms":  float(np.percentile(times, 95)),
        "throughput_fps":  float(1000.0 / np.mean(times)),
        "predictions":   all_preds,
        "labels":        all_labels,
        "probabilities": all_probs,
    }

    print(f"\n[NPU Results]")
    print(f"  Device          : {ov_device}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (weighted)   : {f1:.4f}")
    print(f"  Mean latency    : {np.mean(times):.1f} ms/subject")
    print(f"  P95  latency    : {np.percentile(times, 95):.1f} ms")
    print(f"  Throughput      : {1000/np.mean(times):.1f} subjects/sec")

    out = RESULTS_DIR / f"{model_name}_npu_{ov_device.replace('.','_')}_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  Results saved   : {out}")
    return results


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ===========================================================================
# Benchmark: compare latency across all devices
# ===========================================================================

def benchmark_devices(model_name: str, n_runs: int = 20) -> None:
    """
    Run a timing benchmark on a single random volume across all
    available OpenVINO devices and print a comparison table.
    """
    import openvino as ov
    core  = ov.Core()
    avail = core.available_devices

    xml_path = MODELS_DIR / f"{model_name}_ov" / "model.xml"
    if not xml_path.exists():
        print(f"[Benchmark] OpenVINO IR not found. Train first.")
        return

    dummy = np.random.rand(1, INPUT_CHANNELS, *SPATIAL_SIZE).astype(np.float32)
    ov_model = core.read_model(str(xml_path))

    print(f"\n{'='*55}")
    print(f"  Benchmark: {model_name}  ({n_runs} runs per device)")
    print(f"  Input shape: (1, {INPUT_CHANNELS}, {SPATIAL_SIZE[0]}, {SPATIAL_SIZE[1]}, {SPATIAL_SIZE[2]})")
    print(f"{'='*55}")
    print(f"  {'Device':<12}  {'Mean (ms)':>10}  {'P95 (ms)':>10}  {'FPS':>8}")
    print(f"  {'-'*44}")

    for dev in avail:
        try:
            compiled = core.compile_model(ov_model, dev)
            req      = compiled.create_infer_request()
            # Warmup
            for _ in range(3):
                req.infer({"image": dummy})
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                req.infer({"image": dummy})
                times.append((time.perf_counter() - t0) * 1000)
            mean_t = np.mean(times)
            p95_t  = np.percentile(times, 95)
            fps    = 1000.0 / mean_t
            print(f"  {dev:<12}  {mean_t:>10.1f}  {p95_t:>10.1f}  {fps:>8.2f}")
        except Exception as e:
            print(f"  {dev:<12}  FAILED: {e}")

    print(f"{'='*55}\n")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2023 NPU Inference via OpenVINO")
    parser.add_argument("--model", required=True, help="Model name (from training registry)")
    parser.add_argument("--task", default="GLI", choices=["GLI", "MEN", "PED", "ALL"])
    parser.add_argument("--label-col", default="label_grade",
                        choices=["label", "label_grade"])
    parser.add_argument("--ov-device", default=None,
                        help="OpenVINO device: NPU | GPU.0 | GPU.1 | CPU (auto-select if omitted)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark across all devices")
    parser.add_argument("--benchmark-runs", type=int, default=20)
    args = parser.parse_args()

    if args.benchmark:
        benchmark_devices(args.model, n_runs=args.benchmark_runs)
    else:
        run_npu_inference(
            model_name=args.model,
            task=args.task,
            label_col=args.label_col,
            ov_device=args.ov_device,
        )
