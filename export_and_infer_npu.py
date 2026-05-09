"""
export_and_infer_npu.py
-----------------------
1. Loads the trained ResNet18 checkpoint
2. Exports PyTorch -> ONNX -> OpenVINO IR  (runs on CPU, doesn't touch GPU)
3. Compiles and runs full test-set inference on Intel AI Boost NPU
4. Falls back to GPU.0 (Intel iGPU) or CPU if NPU not available
"""
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows to avoid emoji encoding crashes in PyTorch ONNX logs
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import torch

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_TEST_MANIFEST, INPUT_CHANNELS, MODELS_DIR,
    REGISTRY_PATH, RESULTS_DIR, SPATIAL_SIZE,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model

MODEL_NAME = "resnet18_medicalnet"
CKPT_PATH  = MODELS_DIR / "resnet18_medicalnet_lgg_hgg_3d.pt"
ONNX_PATH  = MODELS_DIR / "resnet18_medicalnet.onnx"
OV_DIR     = MODELS_DIR / "resnet18_medicalnet_ov"
OV_XML     = OV_DIR / "model.xml"

DEVICE_PRIORITY = ["GPU.0", "NPU", "GPU.1", "CPU"]   # GPU.0=Intel Arc iGPU first (NPU crashes on 96^3 volumes)


# ── Step 1: Export to ONNX ──────────────────────────────────────────────────

def export_onnx():
    if ONNX_PATH.exists():
        print(f"[ONNX] Already exported: {ONNX_PATH}")
        return
    print("[ONNX] Loading PyTorch checkpoint...")
    ckpt   = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model  = get_model(MODEL_NAME, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().cpu()
    dummy  = torch.zeros(1, INPUT_CHANNELS, *SPATIAL_SIZE)
    print(f"[ONNX] Exporting (input: {tuple(dummy.shape)})...")

    # Use legacy export API (avoid onnxscript / emoji crash on Python 3.14 + Windows)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(ONNX_PATH),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}},
            opset_version=14,        # opset 14 uses legacy path, no onnxscript needed
            do_constant_folding=True,
            export_params=True,
        )
    import onnx
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    size_mb = ONNX_PATH.stat().st_size / 1024**2
    print(f"[ONNX] Saved: {ONNX_PATH.name}  ({size_mb:.1f} MB)")


# ── Step 2: Convert ONNX → OpenVINO IR (fp16) ────────────────────────────────

def export_openvino():
    if OV_XML.exists():
        print(f"[OV] Already exported: {OV_XML}")
        return
    import openvino as ov
    OV_DIR.mkdir(exist_ok=True)
    print("[OV] Converting ONNX → OpenVINO IR (fp16)...")
    core     = ov.Core()
    ov_model = core.read_model(str(ONNX_PATH))
    ov.save_model(ov_model, str(OV_XML), compress_to_fp16=True)
    bin_size = (OV_DIR / "model.bin").stat().st_size / 1024**2
    print(f"[OV] IR saved: {OV_XML}  ({bin_size:.1f} MB weights)")


# ── Step 3: Select best NPU/iGPU device ──────────────────────────────────────

def pick_device():
    import openvino as ov
    core  = ov.Core()
    avail = core.available_devices
    print(f"[OV] Available devices: {avail}")
    for d in DEVICE_PRIORITY:
        if d in avail:
            name = core.get_property(d, "FULL_DEVICE_NAME")
            print(f"[OV] Selected: {d}  ({name})")
            return d
    return "CPU"


# ── Step 4: Run inference on test set ────────────────────────────────────────

def run_inference(ov_device: str):
    import openvino as ov
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    core     = ov.Core()
    ov_model = core.read_model(str(OV_XML))

    # NPU requires fully static input shapes (no dynamic batch dimension)
    static_shape = [1, INPUT_CHANNELS] + list(SPATIAL_SIZE)
    ov_model.reshape({"image": static_shape})
    print(f"[OV] Static shape set: {static_shape}")

    # Try preferred device, fall back progressively
    fallback = [ov_device] + [d for d in DEVICE_PRIORITY if d != ov_device]
    compiled  = None
    for dev in fallback:
        if dev not in core.available_devices:
            continue
        try:
            cfg = {"PERFORMANCE_HINT": "THROUGHPUT"} if dev == "NPU" else \
                  {"PERFORMANCE_HINT": "LATENCY"} if dev.startswith("GPU") else {}
            compiled  = core.compile_model(ov_model, dev, cfg)
            ov_device = dev
            dname = core.get_property(dev, "FULL_DEVICE_NAME")
            print(f"[OV] Running on: {dev}  ({dname})")
            break
        except Exception as e:
            print(f"[OV] {dev} failed: {str(e)[:100]} — trying next device")

    if compiled is None:
        raise RuntimeError("All OpenVINO devices failed.")

    infer_req = compiled.create_infer_request()

    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test[df_test["task"] == "GLI"].reset_index(drop=True)
    print(f"[OV] Test subjects: {len(df_test)}")

    test_ds = BraTS2023Dataset(df_test, transforms=get_val_transforms(),
                               label_col="label_grade")

    preds, labels, probs, latencies = [], [], [], []
    print(f"\n[OV] Running inference...\n")

    for i in range(len(test_ds)):
        sample = test_ds[i]
        img    = sample["image"].unsqueeze(0).numpy()
        label  = int(sample["label"].item())

        t0 = time.perf_counter()
        infer_req.infer({"image": img})
        logits  = infer_req.get_output_tensor(0).data[0]
        elapsed = (time.perf_counter() - t0) * 1000

        e    = np.exp(logits - logits.max())
        prob = e / e.sum()
        pred = int(np.argmax(prob))

        preds.append(pred)
        labels.append(label)
        probs.append(prob.tolist())
        latencies.append(elapsed)

        if (i + 1) % 25 == 0 or i == 0:
            grade = "LGG" if label == 0 else "HGG"
            print(f"  [{i+1:3d}/{len(test_ds)}] pred={'LGG' if pred==0 else 'HGG'} "
                  f"true={grade}  conf={prob.max():.3f}  lat={elapsed:.1f}ms")

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(labels, [p[1] for p in probs])
    except Exception:
        auc = 0.0

    print(f"\n{'='*55}")
    print(f"  {ov_device} Inference Results — resnet18_medicalnet")
    print(f"{'='*55}")
    print(f"  Subjects       : {len(df_test)}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 (weighted)  : {f1:.4f}")
    print(f"  AUC            : {auc:.4f}")
    print(f"  Mean latency   : {np.mean(latencies):.1f} ms/subject")
    print(f"  P95  latency   : {np.percentile(latencies, 95):.1f} ms")
    print(f"  Throughput     : {1000/np.mean(latencies):.1f} subjects/sec")
    print(f"{'='*55}")

    result = {
        "model": MODEL_NAME, "device": ov_device,
        "n_subjects": len(df_test),
        "accuracy": float(acc), "f1_weighted": float(f1), "auc": float(auc),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms":  float(np.percentile(latencies, 95)),
        "throughput_fps":  float(1000 / np.mean(latencies)),
        "predictions": preds, "labels": labels, "probabilities": probs,
    }
    out = RESULTS_DIR / f"resnet18_{ov_device.replace('.','_')}_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Results saved: {out.name}")
    return result



# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print(" ResNet18 NPU Export + Inference")
    print(" GPU is training DenseNet/Swin in background")
    print("="*55)
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    export_onnx()
    export_openvino()
    device = pick_device()
    run_inference(device)
