"""
run_multi_model_temperature_scaling.py
---------------------------------------
Post-hoc Temperature Scaling experiment across all 9 canonical baseline checkpoints
(Protocols A, B, C across ResNet18, DenseNet121, SwinUNETR) + Focal Loss reference.

Strict Scientific & Data-Isolation Rules:
1. NO training or retraining of any model.
2. NO modification of checkpoints or existing baseline files.
3. Temperature T* is fitted STRICTLY on the 187 validation subjects.
4. T* is frozen before evaluating on the 188 locked test subjects.
5. Invariance checks verify that ROC-AUC and tau=0.50 decisions remain unchanged.
6. All artifacts are saved in an isolated directory:
   results/posthoc_calibration_comparison/
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_VAL_MANIFEST,
    BRATS2023_TEST_MANIFEST,
    RESULTS_DIR,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model

OUT_DIR = RESULTS_DIR / "posthoc_calibration_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR = OUT_DIR / "calibrated_predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_DIR = RESULTS_DIR / "protocol_archives"

# Definition of the 9 baseline models + 1 Focal reference
MODEL_DEFINITIONS = [
    {
        "id": "resnet18_protocolA",
        "model_key": "resnet18_medicalnet",
        "display_name": "ResNet18 MedicalNet",
        "protocol": "Protocol A",
        "protocol_key": "protocol_A",
        "loss_type": "Standard Unweighted CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_A" / "resnet18_medicalnet_lgg_hgg_3d.pt",
    },
    {
        "id": "densenet121_protocolA",
        "model_key": "densenet121_3d",
        "display_name": "DenseNet121 3D",
        "protocol": "Protocol A",
        "protocol_key": "protocol_A",
        "loss_type": "Standard Unweighted CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_A" / "densenet121_3d_lgg_hgg_3d.pt",
    },
    {
        "id": "swin_protocolA",
        "model_key": "swin_transformer_3d",
        "display_name": "SwinUNETR 3D",
        "protocol": "Protocol A",
        "protocol_key": "protocol_A",
        "loss_type": "Standard Unweighted CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_A" / "swin_transformer_3d_lgg_hgg_3d.pt",
    },
    {
        "id": "resnet18_protocolB",
        "model_key": "resnet18_medicalnet",
        "display_name": "ResNet18 MedicalNet",
        "protocol": "Protocol B",
        "protocol_key": "protocol_B",
        "loss_type": "Inverse-Frequency CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_B" / "resnet18_medicalnet_lgg_hgg_3d.pt",
    },
    {
        "id": "densenet121_protocolB",
        "model_key": "densenet121_3d",
        "display_name": "DenseNet121 3D",
        "protocol": "Protocol B",
        "protocol_key": "protocol_B",
        "loss_type": "Inverse-Frequency CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_B" / "densenet121_3d_lgg_hgg_3d.pt",
    },
    {
        "id": "swin_protocolB",
        "model_key": "swin_transformer_3d",
        "display_name": "SwinUNETR 3D",
        "protocol": "Protocol B",
        "protocol_key": "protocol_B",
        "loss_type": "Inverse-Frequency CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_B" / "swin_transformer_3d_lgg_hgg_3d.pt",
    },
    {
        "id": "resnet18_protocolC",
        "model_key": "resnet18_medicalnet",
        "display_name": "ResNet18 MedicalNet",
        "protocol": "Protocol C",
        "protocol_key": "protocol_C",
        "loss_type": "Adjusted Minority CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_C" / "resnet18_medicalnet_lgg_hgg_3d.pt",
    },
    {
        "id": "densenet121_protocolC",
        "model_key": "densenet121_3d",
        "display_name": "DenseNet121 3D",
        "protocol": "Protocol C",
        "protocol_key": "protocol_C",
        "loss_type": "Adjusted Minority CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_C" / "densenet121_3d_lgg_hgg_3d.pt",
    },
    {
        "id": "swin_protocolC",
        "model_key": "swin_transformer_3d",
        "display_name": "SwinUNETR 3D",
        "protocol": "Protocol C",
        "protocol_key": "protocol_C",
        "loss_type": "Adjusted Minority CE",
        "ckpt_path": ARCHIVE_DIR / "protocol_C" / "swin_transformer_3d_lgg_hgg_3d.pt",
    },
    {
        "id": "resnet18_focal",
        "model_key": "resnet18_medicalnet",
        "display_name": "ResNet18 MedicalNet (Focal)",
        "protocol": "Focal Loss",
        "protocol_key": "focal",
        "loss_type": "Focal Loss (alpha=1.0, gamma=2.0)",
        "ckpt_path": BASE_DIR / "models" / "resnet18_medicalnet_focal_lgg_hgg_3d.pt",
    },
]


def compute_nll(logits: np.ndarray, labels: np.ndarray, temp: float = 1.0) -> float:
    """Computes multi-class Negative Log-Likelihood (CrossEntropyLoss)."""
    scaled_logits = logits / temp
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    nll = -log_probs[np.arange(len(labels)), labels].mean()
    return float(nll)


def compute_calibration_metrics(y_true: np.ndarray, p_hgg: np.ndarray, p_lgg: np.ndarray, n_bins: int = 10):
    """Computes confidence ECE/MCE, probability ECE/MCE for HGG and LGG, and Brier score."""
    conf = np.maximum(p_hgg, p_lgg)
    pred = (p_hgg >= 0.5).astype(int)
    acc = (pred == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # 1. Confidence ECE
    ece_conf, mce_conf = 0.0, 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (conf > low) & (conf <= high) if i > 0 else (conf >= low) & (conf <= high)
        n_b = int(mask.sum())
        if n_b > 0:
            gap = abs(float(acc[mask].mean()) - float(conf[mask].mean()))
            ece_conf += gap * (n_b / len(y_true))
            mce_conf = max(mce_conf, gap)

    # 2. HGG Probability ECE
    ece_hgg, mce_hgg = 0.0, 0.0
    hgg_bins = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (p_hgg > low) & (p_hgg <= high) if i > 0 else (p_hgg >= low) & (p_hgg <= high)
        n_b = int(mask.sum())
        if n_b > 0:
            mean_p = float(p_hgg[mask].mean())
            frac_pos = float((y_true[mask] == 1).mean())
            gap = abs(frac_pos - mean_p)
            ece_hgg += gap * (n_b / len(y_true))
            mce_hgg = max(mce_hgg, gap)
            hgg_bins.append({
                "bin_range": f"({low:.1f}, {high:.1f}]" if i > 0 else f"[{low:.1f}, {high:.1f}]",
                "n_samples": n_b,
                "mean_p": round(mean_p, 4),
                "frac_pos": round(frac_pos, 4),
                "gap": round(gap, 4),
            })

    # 3. LGG Probability ECE
    ece_lgg, mce_lgg = 0.0, 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (p_lgg > low) & (p_lgg <= high) if i > 0 else (p_lgg >= low) & (p_lgg <= high)
        n_b = int(mask.sum())
        if n_b > 0:
            mean_p = float(p_lgg[mask].mean())
            frac_pos = float((y_true[mask] == 0).mean())
            gap = abs(frac_pos - mean_p)
            ece_lgg += gap * (n_b / len(y_true))
            mce_lgg = max(mce_lgg, gap)

    brier_score = float(np.mean((p_hgg - y_true) ** 2))

    return {
        "brier_score": round(brier_score, 4),
        "confidence_ece": round(float(ece_conf), 4),
        "confidence_mce": round(float(mce_conf), 4),
        "hgg_prob_ece": round(float(ece_hgg), 4),
        "hgg_prob_mce": round(float(mce_hgg), 4),
        "lgg_prob_ece": round(float(ece_lgg), 4),
        "lgg_prob_mce": round(float(mce_lgg), 4),
        "hgg_bins": hgg_bins,
    }


def compute_metrics_at_threshold(y_true: np.ndarray, p_lgg: np.ndarray, tau: float) -> dict:
    """Evaluates decision boundary tau on P(LGG): predict LGG (0) if p_lgg >= tau else HGG (1)."""
    y_pred = (p_lgg < tau).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    lgg_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    hgg_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "threshold_tau_lgg": round(tau, 2),
        "threshold_theta_hgg": round(1.0 - tau, 2),
        "lgg_recall": round(lgg_rec, 4),
        "hgg_recall": round(hgg_rec, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_m, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    start_time = time.time()
    print("=" * 90)
    print("MULTI-MODEL POST-HOC TEMPERATURE SCALING & CALIBRATION BENCHMARK")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # ---------------------------------------------------------------------------
    # Step 1: Load and Pre-Cache Validation & Locked-Test 3D Tensors in RAM
    # ---------------------------------------------------------------------------
    print("\n[Step 1] Loading and Pre-Caching Cohorts in RAM for fast forward passes...")
    
    # Validation cohort (n=187)
    df_val_raw = load_manifest(BRATS2023_VAL_MANIFEST, label_col="label_grade")
    df_val = df_val_raw[df_val_raw["task"] == "GLI"].reset_index(drop=True)
    ds_val = BraTS2023Dataset(df_val, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    val_loader = DataLoader(ds_val, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    t0_cache = time.time()
    val_batches = []
    for batch in val_loader:
        val_batches.append({
            "image": batch["image"],
            "label": batch["label"],
            "subject_id": batch.get("subject_id", [])
        })
    val_y_true = np.concatenate([b["label"].numpy() for b in val_batches]).astype(int)
    val_sub_ids = [s for b in val_batches for s in b["subject_id"]]
    print(f"  Cached {len(df_val)} validation subjects ({int((val_y_true==0).sum())} LGG, {int((val_y_true==1).sum())} HGG) in {time.time()-t0_cache:.2f}s")

    # Locked test cohort (n=188)
    df_test_raw = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test_raw[df_test_raw["task"] == "GLI"].reset_index(drop=True)
    ds_test = BraTS2023Dataset(df_test, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    test_loader = DataLoader(ds_test, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    t0_cache2 = time.time()
    test_batches = []
    for batch in test_loader:
        test_batches.append({
            "image": batch["image"],
            "label": batch["label"],
            "subject_id": batch.get("subject_id", [])
        })
    test_y_true = np.concatenate([b["label"].numpy() for b in test_batches]).astype(int)
    test_sub_ids = [s for b in test_batches for s in b["subject_id"]]
    print(f"  Cached {len(df_test)} locked-test subjects ({int((test_y_true==0).sum())} LGG, {int((test_y_true==1).sum())} HGG) in {time.time()-t0_cache2:.2f}s")

    # ---------------------------------------------------------------------------
    # Step 2: Iterate across all 10 models (9 baseline + 1 Focal reference)
    # ---------------------------------------------------------------------------
    print("\n[Step 2] Executing Forward Passes & Validation Temperature Optimization...")

    summary_records = []
    temperature_params = {}
    threshold_sweep_records = []
    reliability_diagram_data = {}

    thresholds_arr = np.arange(0.10, 0.9001, 0.05)

    for m_info in MODEL_DEFINITIONS:
        m_id = m_info["id"]
        m_key = m_info["model_key"]
        d_name = m_info["display_name"]
        proto = m_info["protocol"]
        ckpt_path = m_info["ckpt_path"]

        print(f"\n---> Processing Model: [{proto}] {d_name} ({ckpt_path.name})")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model = get_model(m_key, num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        # 1. Forward Pass on Validation Split ONLY
        val_logits_list = []
        with torch.no_grad():
            for b in val_batches:
                imgs = b["image"].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    raw_logits = model(imgs)
                val_logits_list.append(raw_logits.float().cpu().numpy())
        val_logits = np.concatenate(val_logits_list, axis=0)

        # 2. Fit T* strictly on Validation NLL
        val_nll_initial = compute_nll(val_logits, val_y_true, temp=1.0)
        
        def val_nll_objective(t_scalar):
            return compute_nll(val_logits, val_y_true, temp=t_scalar)

        opt_res = minimize_scalar(val_nll_objective, bounds=(0.01, 10.0), method="bounded")
        FROZEN_T = float(opt_res.x)
        val_nll_cal = float(opt_res.fun)

        val_p_raw = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
        val_p_cal = torch.softmax(torch.from_numpy(val_logits / FROZEN_T), dim=1).numpy()
        val_cal_raw = compute_calibration_metrics(val_y_true, val_p_raw[:, 1], val_p_raw[:, 0])
        val_cal_opt = compute_calibration_metrics(val_y_true, val_p_cal[:, 1], val_p_cal[:, 0])

        print(f"     Optimal T* = {FROZEN_T:.4f} (Val NLL: {val_nll_initial:.4f} -> {val_nll_cal:.4f}, Val ECE: {val_cal_raw['confidence_ece']:.4f} -> {val_cal_opt['confidence_ece']:.4f})")

        # 3. Forward Pass on Locked Test Split
        test_logits_list = []
        with torch.no_grad():
            for b in test_batches:
                imgs = b["image"].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=True):
                    raw_logits = model(imgs)
                test_logits_list.append(raw_logits.float().cpu().numpy())
        test_logits = np.concatenate(test_logits_list, axis=0)

        # 4. Apply Frozen T* to Test Logits
        test_logits_64 = test_logits.astype(np.float64)
        test_nll_raw = compute_nll(test_logits_64, test_y_true, temp=1.0)
        test_nll_cal = compute_nll(test_logits_64, test_y_true, temp=FROZEN_T)

        test_p_raw = torch.softmax(torch.from_numpy(test_logits_64), dim=1).numpy()
        test_p_cal = torch.softmax(torch.from_numpy(test_logits_64 / FROZEN_T), dim=1).numpy()

        test_cal_raw = compute_calibration_metrics(test_y_true, test_p_raw[:, 1], test_p_raw[:, 0])
        test_cal_opt = compute_calibration_metrics(test_y_true, test_p_cal[:, 1], test_p_cal[:, 0])

        # Classification metrics at standard tau=0.50
        pred_raw_05 = (test_p_raw[:, 1] >= 0.5).astype(int)
        pred_cal_05 = (test_p_cal[:, 1] >= 0.5).astype(int)

        # Invariance Assertions
        auc_raw = float(roc_auc_score(test_y_true, test_p_raw[:, 1]))
        auc_cal = float(roc_auc_score(test_y_true, test_p_cal[:, 1]))
        assert np.isclose(auc_raw, auc_cal, atol=1e-4), f"Invariance broken for {m_id}: AUC changed ({auc_raw:.5f} vs {auc_cal:.5f})!"
        assert np.array_equal(pred_raw_05, pred_cal_05), f"Invariance broken for {m_id}: tau=0.50 predictions changed!"

        acc_05 = float(accuracy_score(test_y_true, pred_cal_05))
        f1_m_05 = float(f1_score(test_y_true, pred_cal_05, average="macro", zero_division=0))
        bal_acc_05 = float(balanced_accuracy_score(test_y_true, pred_cal_05))

        cm = confusion_matrix(test_y_true, pred_cal_05, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        lgg_rec_05 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        hgg_rec_05 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Save Calibrated Predictions CSV
        cal_conf = np.maximum(test_p_cal[:, 0], test_p_cal[:, 1])
        p0c = np.clip(test_p_cal[:, 0], 1e-12, 1.0)
        p1c = np.clip(test_p_cal[:, 1], 1e-12, 1.0)
        cal_unc = -(p0c * np.log2(p0c) + p1c * np.log2(p1c))

        df_pred_export = pd.DataFrame({
            "subject_id": test_sub_ids,
            "true_label": test_y_true,
            "logit_lgg": test_logits[:, 0],
            "logit_hgg": test_logits[:, 1],
            "raw_prob_lgg": test_p_raw[:, 0],
            "raw_prob_hgg": test_p_raw[:, 1],
            "cal_prob_lgg": test_p_cal[:, 0],
            "cal_prob_hgg": test_p_cal[:, 1],
            "predicted_label_05": pred_cal_05,
            "cal_confidence": cal_conf,
            "cal_uncertainty": cal_unc,
        })
        pred_csv_name = f"{m_id}_calibrated_predictions.csv"
        df_pred_export.to_csv(PRED_DIR / pred_csv_name, index=False)

        # Store Summary Record
        summary_records.append({
            "model_id": m_id,
            "display_name": d_name,
            "protocol": proto,
            "loss_type": m_info["loss_type"],
            "frozen_T": round(FROZEN_T, 4),
            "val_nll_raw": round(val_nll_initial, 4),
            "val_nll_cal": round(val_nll_cal, 4),
            "val_ece_raw": round(val_cal_raw["confidence_ece"], 4),
            "val_ece_cal": round(val_cal_opt["confidence_ece"], 4),
            "test_nll_raw": round(test_nll_raw, 4),
            "test_nll_cal": round(test_nll_cal, 4),
            "delta_test_nll": round(test_nll_cal - test_nll_raw, 4),
            "test_brier_raw": round(test_cal_raw["brier_score"], 4),
            "test_brier_cal": round(test_cal_opt["brier_score"], 4),
            "delta_test_brier": round(test_cal_opt["brier_score"] - test_cal_raw["brier_score"], 4),
            "test_ece_raw": round(test_cal_raw["confidence_ece"], 4),
            "test_ece_cal": round(test_cal_opt["confidence_ece"], 4),
            "delta_test_ece": round(test_cal_opt["confidence_ece"] - test_cal_raw["confidence_ece"], 4),
            "test_prob_ece_raw": round(test_cal_raw["hgg_prob_ece"], 4),
            "test_prob_ece_cal": round(test_cal_opt["hgg_prob_ece"], 4),
            "test_mce_raw": round(test_cal_raw["confidence_mce"], 4),
            "test_mce_cal": round(test_cal_opt["confidence_mce"], 4),
            "test_auc": round(auc_cal, 4),
            "test_accuracy_05": round(acc_05, 4),
            "test_macro_f1_05": round(f1_m_05, 4),
            "test_balanced_acc_05": round(bal_acc_05, 4),
            "test_lgg_recall_05": round(lgg_rec_05, 4),
            "test_hgg_recall_05": round(hgg_rec_05, 4),
        })

        temperature_params[m_id] = {
            "model_name": d_name,
            "protocol": proto,
            "loss_type": m_info["loss_type"],
            "frozen_T": round(FROZEN_T, 6),
            "validation_fit": {
                "n_samples": len(val_y_true),
                "val_nll_initial": round(val_nll_initial, 6),
                "val_nll_calibrated": round(val_nll_cal, 6),
                "val_ece_raw": val_cal_raw["confidence_ece"],
                "val_ece_cal": val_cal_opt["confidence_ece"],
                "val_brier_raw": val_cal_raw["brier_score"],
                "val_brier_cal": val_cal_opt["brier_score"],
            },
            "locked_test_evaluation": {
                "n_samples": len(test_y_true),
                "test_nll_raw": round(test_nll_raw, 6),
                "test_nll_calibrated": round(test_nll_cal, 6),
                "test_ece_raw": test_cal_raw["confidence_ece"],
                "test_ece_calibrated": test_cal_opt["confidence_ece"],
                "test_brier_raw": test_cal_raw["brier_score"],
                "test_brier_calibrated": test_cal_opt["brier_score"],
                "auc_invariant": round(auc_cal, 4),
            }
        }

        reliability_diagram_data[m_id] = {
            "display_name": f"{d_name} ({proto})",
            "raw_hgg_bins": test_cal_raw["hgg_bins"],
            "cal_hgg_bins": test_cal_opt["hgg_bins"],
            "ece_raw": test_cal_raw["confidence_ece"],
            "ece_cal": test_cal_opt["confidence_ece"],
        }

        # 5. Threshold Analysis on Calibrated Probabilities
        for tau in thresholds_arr:
            m_th = compute_metrics_at_threshold(test_y_true, test_p_cal[:, 0], tau)
            threshold_sweep_records.append({
                "model_id": m_id,
                "display_name": d_name,
                "protocol": proto,
                **m_th
            })

        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Step 3: Export Master CSVs and JSONs
    # ---------------------------------------------------------------------------
    print("\n[Step 3] Exporting Summary Tables and Manifests...")

    # 1. multi_model_calibration_summary.csv
    df_summary = pd.DataFrame(summary_records)
    summary_csv_path = OUT_DIR / "multi_model_calibration_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"  Saved: {summary_csv_path.name}")

    # 2. temperature_parameters.json
    param_json_path = OUT_DIR / "temperature_parameters.json"
    param_json_path.write_text(json.dumps(temperature_params, indent=2))
    print(f"  Saved: {param_json_path.name}")

    # 3. multi_model_threshold_comparison.csv
    df_threshold_all = pd.DataFrame(threshold_sweep_records)
    threshold_csv_path = OUT_DIR / "multi_model_threshold_comparison.csv"
    df_threshold_all.to_csv(threshold_csv_path, index=False)
    print(f"  Saved: {threshold_csv_path.name}")

    # 4. experiment_metadata.json
    metadata = {
        "experiment_title": "Multi-Model Post-Hoc Temperature Scaling Comparison",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohorts": {
            "validation": {"manifest": str(BRATS2023_VAL_MANIFEST), "n_subjects": 187, "lgg": 10, "hgg": 177},
            "locked_test": {"manifest": str(BRATS2023_TEST_MANIFEST), "n_subjects": 188, "lgg": 12, "hgg": 176},
        },
        "leakage_controls": {
            "val_only_fitting": "Temperature parameter T* optimized strictly on validation NLL.",
            "test_isolation": "Locked test set used strictly in read-only inference with frozen T*.",
            "invariance_verified": "Verified that ROC-AUC, prediction rank order, and tau=0.50 decisions are preserved 100%.",
        },
        "models_evaluated": len(MODEL_DEFINITIONS),
        "summary_results": summary_records,
    }
    meta_json_path = OUT_DIR / "experiment_metadata.json"
    meta_json_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Saved: {meta_json_path.name}")

    # ---------------------------------------------------------------------------
    # Step 4: Generate Publication-Quality Multi-Panel Reliability Diagrams
    # ---------------------------------------------------------------------------
    print("\n[Step 4] Generating Multi-Model Reliability Diagrams...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=300)
    fig.suptitle("Locked-Test Multi-Model Reliability Diagrams: Raw vs Post-Hoc Temperature Scaled", fontsize=15, fontweight="bold", y=0.995)

    protocols = ["Protocol A", "Protocol B", "Protocol C"]
    models = ["resnet18", "densenet121", "swin"]
    model_titles = ["ResNet18 MedicalNet", "DenseNet121 3D", "SwinUNETR 3D"]

    for row_idx, proto in enumerate(protocols):
        proto_key = "protocolA" if proto == "Protocol A" else ("protocolB" if proto == "Protocol B" else "protocolC")
        for col_idx, m_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            m_id = f"{m_name}_{proto_key}"
            
            # Diagonal line
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration", linewidth=1.2)

            if m_id in reliability_diagram_data:
                d = reliability_diagram_data[m_id]
                rx = [b["mean_p"] for b in d["raw_hgg_bins"]]
                ry = [b["frac_pos"] for b in d["raw_hgg_bins"]]
                cx = [b["mean_p"] for b in d["cal_hgg_bins"]]
                cy = [b["frac_pos"] for b in d["cal_hgg_bins"]]

                ax.plot(rx, ry, marker="s", color="#d95f02", linestyle="--", linewidth=1.6, label=f"Raw (ECE={d['ece_raw']:.3f})")
                ax.plot(cx, cy, marker="o", color="#2b5c8f", linewidth=2.0, label=f"Calibrated (ECE={d['ece_cal']:.3f})")

            ax.set_title(f"{model_titles[col_idx]} — {proto}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted P(HGG)", fontsize=9.5)
            ax.set_ylabel("Observed Fraction HGG", fontsize=9.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    diagram_png_path = OUT_DIR / "multi_model_reliability_diagrams.png"
    plt.savefig(diagram_png_path, dpi=300)
    plt.close()
    print(f"  Saved Plot: {diagram_png_path.name}")

    # ---------------------------------------------------------------------------
    # Step 5: Generate README.md Documentation
    # ---------------------------------------------------------------------------
    readme_content = f"""# Post-Hoc Temperature Scaling Comparison Experiment

## Overview
This experiment evaluates post-hoc temperature scaling across all 9 canonical BraTS 2023 GLI grading checkpoints (Protocols A, B, C across ResNet18, DenseNet121, SwinUNETR) plus the Focal Loss ResNet18 intervention.

## Strict Leakage Controls
- **Validation-Only Fitting:** Temperature parameter $T^*$ was optimized strictly by minimizing Negative Log-Likelihood (NLL) on the 187 validation subjects.
- **Locked Test Isolation:** The 188 locked test subjects were processed strictly in read-only forward mode using the frozen $T^*$ parameters.
- **Mathematical Invariance:** Verified that relative rank orderings, ROC-AUC, and binary classifications at $\\tau=0.50$ remain invariant under scalar temperature scaling.

## Cohort Details
- **Validation Split:** 187 subjects (10 LGG, 177 HGG) from `processed_dataset/brats2023/val_manifest.json`
- **Locked Test Split:** 188 subjects (12 LGG, 176 HGG) from `processed_dataset/brats2023/test_manifest.json`
- **Deterministic Transforms:** Standard `get_val_transforms()` ($4 \\times 96 \\times 96 \\times 96$ volume normalization).

## Generated Files
- `multi_model_calibration_summary.csv`: Master metrics comparison table.
- `temperature_parameters.json`: Optimized frozen $T^*$ parameters and convergence metadata.
- `calibrated_predictions/`: Per-model subject-level calibrated predictions CSVs.
- `multi_model_reliability_diagrams.png`: 9-panel publication reliability diagram.
- `multi_model_threshold_comparison.csv`: Full threshold sweep table across all calibrated models.
- `experiment_metadata.json`: Complete machine-readable experiment manifest.
"""
    readme_path = OUT_DIR / "README.md"
    readme_path.write_text(readme_content)
    print(f"  Saved: {readme_path.name}")

    # ---------------------------------------------------------------------------
    # Step 6: Console Summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - start_time
    print("\n" + "=" * 105)
    print(f"EXPERIMENT COMPLETE — Total Runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 105)
    print(df_summary[["model_id", "protocol", "frozen_T", "test_nll_raw", "test_nll_cal", "test_ece_raw", "test_ece_cal", "test_brier_raw", "test_brier_cal", "test_auc", "test_lgg_recall_05", "test_hgg_recall_05"]].to_string(index=False))


if __name__ == "__main__":
    main()
