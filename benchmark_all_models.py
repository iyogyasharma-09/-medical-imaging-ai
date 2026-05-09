"""
benchmark_all_models.py
-----------------------
Exports all 3 trained models to ONNX + OpenVINO IR, then benchmarks
inference across all available devices (NPU, iGPU, NVIDIA, CPU).

Models:  resnet18_medicalnet | densenet121_3d | swin_transformer_3d
Devices: GPU.0 (Intel Arc iGPU) | GPU.1 (NVIDIA RTX A1000) | CPU
         NPU attempted but falls back if 3D model too large.
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import torch

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_TEST_MANIFEST, INPUT_CHANNELS, MODELS_DIR, RESULTS_DIR, SPATIAL_SIZE,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CHECKPOINTS = {
    "resnet18_medicalnet":  MODELS_DIR / "resnet18_medicalnet_lgg_hgg_3d.pt",
    "densenet121_3d":       MODELS_DIR / "densenet121_3d_lgg_hgg_3d.pt",
    "swin_transformer_3d":  MODELS_DIR / "swin_transformer_3d_lgg_hgg_3d.pt",
}

# Devices to benchmark (in priority order)
BENCH_DEVICES = ["GPU.0", "GPU.1", "CPU"]   # GPU.0=Intel Arc, GPU.1=NVIDIA RTX


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_onnx(model_name: str, model: torch.nn.Module) -> Path:
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    if onnx_path.exists():
        print(f"  [ONNX] Already exists: {onnx_path.name}")
        return onnx_path

    dummy = torch.zeros(1, INPUT_CHANNELS, *SPATIAL_SIZE)
    model.eval().cpu()
    print(f"  [ONNX] Exporting {model_name}...")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["image"], output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}},
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )
    import onnx
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"  [ONNX] Saved ({onnx_path.stat().st_size/1024**2:.1f} MB)")
    return onnx_path


# ── OpenVINO IR Export ────────────────────────────────────────────────────────

def export_openvino(model_name: str, onnx_path: Path) -> Path:
    import openvino as ov
    ov_dir = MODELS_DIR / f"{model_name}_ov"
    xml    = ov_dir / "model.xml"
    if xml.exists():
        print(f"  [OV]   Already exists: {xml}")
        return xml
    ov_dir.mkdir(exist_ok=True)
    print(f"  [OV]   Converting to OpenVINO IR (fp16)...")
    core     = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    ov.save_model(ov_model, str(xml), compress_to_fp16=True)
    bin_mb = (ov_dir / "model.bin").stat().st_size / 1024**2
    print(f"  [OV]   IR saved ({bin_mb:.1f} MB weights)")
    return xml


# ── Single-device benchmark ───────────────────────────────────────────────────

def benchmark_device(model_name: str, xml_path: Path, device: str,
                     test_ds, n_warmup: int = 3) -> dict:
    import openvino as ov
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    core     = ov.Core()
    ov_model = core.read_model(str(xml_path))

    # Static shape required for NPU; good practice for all devices
    static_shape = [1, INPUT_CHANNELS] + list(SPATIAL_SIZE)
    ov_model.reshape({"image": static_shape})

    cfg = {"PERFORMANCE_HINT": "LATENCY"} if device.startswith("GPU") else {}
    try:
        compiled  = core.compile_model(ov_model, device, cfg)
    except Exception as e:
        return {"device": device, "error": str(e)[:120]}

    infer_req = compiled.create_infer_request()

    # Warm-up
    dummy = np.zeros([1, INPUT_CHANNELS] + list(SPATIAL_SIZE), dtype=np.float32)
    for _ in range(n_warmup):
        infer_req.infer({"image": dummy})

    preds, labels, probs, lats = [], [], [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        img    = sample["image"].unsqueeze(0).numpy().astype(np.float32)
        label  = int(sample["label"].item())

        t0 = time.perf_counter()
        infer_req.infer({"image": img})
        logits  = infer_req.get_output_tensor(0).data[0]
        lat_ms  = (time.perf_counter() - t0) * 1000

        e    = np.exp(logits - logits.max())
        prob = e / e.sum()

        preds.append(int(np.argmax(prob)))
        labels.append(label)
        probs.append(prob.tolist())
        lats.append(lat_ms)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    try:    auc = roc_auc_score(labels, [p[1] for p in probs])
    except: auc = 0.0

    return {
        "device":           device,
        "n_subjects":       len(test_ds),
        "accuracy":         float(acc),
        "f1_weighted":      float(f1),
        "auc":              float(auc),
        "mean_latency_ms":  float(np.mean(lats)),
        "p50_latency_ms":   float(np.percentile(lats, 50)),
        "p95_latency_ms":   float(np.percentile(lats, 95)),
        "throughput_fps":   float(1000 / np.mean(lats)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import openvino as ov
    core   = ov.Core()
    avail  = core.available_devices
    print(f"\nAvailable OpenVINO devices: {avail}")
    bench_devs = [d for d in BENCH_DEVICES if d in avail]
    print(f"Will benchmark on        : {bench_devs}\n")

    # Load test set once
    df_test  = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test  = df_test[df_test["task"] == "GLI"].reset_index(drop=True)
    test_ds  = BraTS2023Dataset(df_test, transforms=get_val_transforms(),
                                label_col="label_grade")
    print(f"Test subjects: {len(test_ds)}\n")

    all_results = {}

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # Load checkpoint
        ckpt  = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = get_model(model_name, num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Export
        onnx_path = export_onnx(model_name, model)
        xml_path  = export_openvino(model_name, onnx_path)

        model_results = {"best_val_acc": ckpt.get("best_val_accuracy"),
                         "epochs":       len(ckpt.get("history", [])),
                         "devices":      {}}

        # Benchmark each device
        for dev in bench_devs:
            dev_name = core.get_property(dev, "FULL_DEVICE_NAME")
            print(f"\n  [{dev}] {dev_name}")
            r = benchmark_device(model_name, xml_path, dev, test_ds)
            model_results["devices"][dev] = r
            if "error" in r:
                print(f"    ERROR: {r['error']}")
            else:
                print(f"    Accuracy   : {r['accuracy']:.4f} ({r['accuracy']*100:.2f}%)")
                print(f"    AUC        : {r['auc']:.4f}")
                print(f"    Latency    : {r['mean_latency_ms']:.1f} ms mean  |  {r['p95_latency_ms']:.1f} ms P95")
                print(f"    Throughput : {r['throughput_fps']:.2f} subjects/sec")

        all_results[model_name] = model_results

    # Save full results
    out = RESULTS_DIR / "benchmark_all_models.json"
    out.write_text(json.dumps(all_results, indent=2))

    # Print summary table
    print(f"\n\n{'='*70}")
    print(f"  BENCHMARK SUMMARY  —  {len(test_ds)} GLI test subjects")
    print(f"{'='*70}")
    hdr = f"  {'Model':<26} {'Device':<8} {'Acc':>7} {'AUC':>7} {'Lat(ms)':>9} {'FPS':>6}"
    print(hdr)
    print(f"  {'-'*64}")
    for mname, mres in all_results.items():
        for dev, r in mres["devices"].items():
            if "error" in r:
                print(f"  {mname:<26} {dev:<8}  ERROR: {r['error'][:30]}")
            else:
                print(f"  {mname:<26} {dev:<8} {r['accuracy']:>7.4f} {r['auc']:>7.4f}"
                      f" {r['mean_latency_ms']:>9.1f} {r['throughput_fps']:>6.2f}")
    print(f"{'='*70}")
    print(f"\n  Full results saved: {out}")


if __name__ == "__main__":
    main()
