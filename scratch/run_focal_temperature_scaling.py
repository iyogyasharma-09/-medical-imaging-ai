"""
run_focal_temperature_scaling.py
---------------------------------
Post-hoc Temperature Scaling experiment for Focal Loss ResNet18 MedicalNet.

Strict Scientific Protocol:
1. Load trained Focal checkpoint (C:\mri\mri-\models\resnet18_medicalnet_focal_lgg_hgg_3d.pt).
2. Forward pass on the 187 VALIDATION subjects only to obtain validation logits and labels.
3. Optimize temperature scalar T > 0 by minimizing Negative Log-Likelihood (NLL) on validation data.
4. Freeze T*.
5. Forward pass on the 188 LOCKED TEST subjects to obtain test logits and labels.
6. Apply frozen T* to scale test logits: z_test_cal = z_test / T*.
7. Compute and evaluate calibrated test probabilities, ECE, Brier score, MCE, and reliability curves.
8. Save all artifacts in C:\mri\mri-\results\focal_locked_test\temperature_scaling\.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

OUT_DIR = RESULTS_DIR / "focal_locked_test" / "temperature_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_calibration_data(y_true: np.ndarray, p_hgg: np.ndarray, p_lgg: np.ndarray, n_bins: int = 10):
    """Computes confidence-based calibration and one-vs-rest probability calibration."""
    conf = np.maximum(p_hgg, p_lgg)
    pred = (p_hgg >= 0.5).astype(int)
    acc = (pred == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # 1. Confidence ECE
    conf_bins = []
    ece_conf = 0.0
    mce_conf = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (conf > low) & (conf <= high) if i > 0 else (conf >= low) & (conf <= high)
        n_b = int(mask.sum())
        if n_b > 0:
            mean_c = float(conf[mask].mean())
            mean_a = float(acc[mask].mean())
            gap = abs(mean_a - mean_c)
            ece_conf += gap * (n_b / len(y_true))
            mce_conf = max(mce_conf, gap)
            conf_bins.append({
                "bin_index": i + 1,
                "bin_range": f"({low:.1f}, {high:.1f}]" if i > 0 else f"[{low:.1f}, {high:.1f}]",
                "bin_lower": round(low, 2),
                "bin_upper": round(high, 2),
                "n_samples": n_b,
                "mean_confidence": round(mean_c, 4),
                "mean_accuracy": round(mean_a, 4),
                "calibration_gap": round(gap, 4),
            })
            
    # 2. HGG Probability Calibration P(HGG) vs y=1
    hgg_bins = []
    ece_hgg = 0.0
    mce_hgg = 0.0
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
                "target_class": "HGG",
                "bin_index": i + 1,
                "bin_range": f"({low:.1f}, {high:.1f}]" if i > 0 else f"[{low:.1f}, {high:.1f}]",
                "bin_lower": round(low, 2),
                "bin_upper": round(high, 2),
                "n_samples": n_b,
                "mean_predicted_prob": round(mean_p, 4),
                "observed_fraction": round(frac_pos, 4),
                "calibration_gap": round(gap, 4),
                "direction": "overconfident" if mean_p > frac_pos else ("underconfident" if mean_p < frac_pos else "perfect"),
            })
            
    # 3. LGG Probability Calibration P(LGG) vs y=0
    lgg_bins = []
    ece_lgg = 0.0
    mce_lgg = 0.0
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
            lgg_bins.append({
                "target_class": "LGG",
                "bin_index": i + 1,
                "bin_range": f"({low:.1f}, {high:.1f}]" if i > 0 else f"[{low:.1f}, {high:.1f}]",
                "bin_lower": round(low, 2),
                "bin_upper": round(high, 2),
                "n_samples": n_b,
                "mean_predicted_prob": round(mean_p, 4),
                "observed_fraction": round(frac_pos, 4),
                "calibration_gap": round(gap, 4),
                "direction": "overconfident" if mean_p > frac_pos else ("underconfident" if mean_p < frac_pos else "perfect"),
            })
            
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
        "lgg_bins": lgg_bins,
        "conf_bins": conf_bins,
    }


def compute_nll(logits: np.ndarray, labels: np.ndarray, temp: float = 1.0) -> float:
    """Computes multi-class Negative Log-Likelihood (CrossEntropyLoss)."""
    scaled_logits = logits / temp
    # Numerically stable log_softmax
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    nll = -log_probs[np.arange(len(labels)), labels].mean()
    return float(nll)


def run_forward_pass(model, loader, device, desc: str):
    """Runs deterministic forward pass and extracts raw unnormalized logits."""
    model.eval()
    subject_ids = []
    y_true_list = []
    logits_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            imgs = batch["image"].to(device)
            labels = batch["label"]
            s_ids = batch.get("subject_id", [f"sub_{i}" for i in range(len(imgs))])

            with torch.amp.autocast(device_type=device.type, enabled=True):
                raw_logits = model(imgs)

            subject_ids.extend(s_ids)
            y_true_list.extend(labels.numpy())
            logits_list.append(raw_logits.float().cpu().numpy())

    y_true = np.array(y_true_list, dtype=int)
    logits = np.concatenate(logits_list, axis=0)  # Shape: (N, 2)
    return subject_ids, y_true, logits


def main():
    start_time = time.time()
    print("=" * 80)
    print("FOCAL LOSS RESNET18: POST-HOC TEMPERATURE SCALING EXPERIMENT")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------------------
    # Step 1: Reconstruct Model from Checkpoint
    # ---------------------------------------------------------------------------
    ckpt_path = BASE_DIR / "models" / "resnet18_medicalnet_focal_lgg_hgg_3d.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing Focal checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"\n[1] Loaded Checkpoint: {ckpt_path.name}")
    print(f"    model_name: {ckpt.get('model_name')}, num_classes: {ckpt.get('num_classes')}")
    print(f"    best_val_macro_f1: {ckpt.get('best_val_macro_f1')}, best_val_lgg_recall: {ckpt.get('best_val_lgg_recall')}")

    model = get_model("resnet18_medicalnet", num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------------------------
    # Step 2: Forward Pass on Validation Split ONLY
    # ---------------------------------------------------------------------------
    print("\n[2] Loading Validation Cohort (for temperature parameter optimization)...")
    df_val_raw = load_manifest(BRATS2023_VAL_MANIFEST, label_col="label_grade")
    df_val = df_val_raw[df_val_raw["task"] == "GLI"].reset_index(drop=True)
    ds_val = BraTS2023Dataset(df_val, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    val_loader = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    n_val_lgg = int((df_val["label_grade"] == 0).sum())
    n_val_hgg = int((df_val["label_grade"] == 1).sum())
    print(f"    Validation subjects: {len(df_val)} (LGG={n_val_lgg}, HGG={n_hgg_val if 'n_hgg_val' in locals() else n_val_hgg})")

    val_sub_ids, val_y_true, val_logits = run_forward_pass(model, val_loader, device, desc="Validation Forward Pass")
    print(f"    Validation logits extracted. Shape: {val_logits.shape}")

    # ---------------------------------------------------------------------------
    # Step 3: Optimize Temperature Parameter T on Validation Split ONLY
    # ---------------------------------------------------------------------------
    print("\n[3] Optimizing Temperature Parameter T on Validation Logits...")
    val_nll_initial = compute_nll(val_logits, val_y_true, temp=1.0)
    print(f"    Initial Validation NLL (T=1.0000): {val_nll_initial:.6f}")

    def nll_objective(t_val):
        return compute_nll(val_logits, val_y_true, temp=t_val)

    # Scipy bounded optimization
    opt_res = minimize_scalar(nll_objective, bounds=(0.01, 10.0), method="bounded")
    optimal_T = float(opt_res.x)
    val_nll_calibrated = float(opt_res.fun)

    print(f"\n    >>> OPTIMAL TEMPERATURE FOUND: T* = {optimal_T:.4f} <<<")
    print(f"    Validation NLL before: {val_nll_initial:.6f}")
    print(f"    Validation NLL after:  {val_nll_calibrated:.6f} (Delta: {val_nll_calibrated - val_nll_initial:+.6f})")
    print(f"    Optimization Status:   Success={opt_res.success}, Iterations={opt_res.nfev}")

    # Validation uncalibrated vs calibrated probabilities
    val_p_raw = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    val_p_cal = torch.softmax(torch.from_numpy(val_logits / optimal_T), dim=1).numpy()

    val_cal_raw = compute_calibration_data(val_y_true, val_p_raw[:, 1], val_p_raw[:, 0], n_bins=10)
    val_cal_opt = compute_calibration_data(val_y_true, val_p_cal[:, 1], val_p_cal[:, 0], n_bins=10)

    print(f"    Val Confidence ECE: {val_cal_raw['confidence_ece']:.4f} -> {val_cal_opt['confidence_ece']:.4f}")
    print(f"    Val Brier Score:    {val_cal_raw['brier_score']:.4f} -> {val_cal_opt['brier_score']:.4f}")

    # Save validation logits & predictions
    df_val_export = pd.DataFrame({
        "subject_id": val_sub_ids,
        "true_label": val_y_true,
        "logit_lgg": val_logits[:, 0],
        "logit_hgg": val_logits[:, 1],
        "raw_prob_lgg": val_p_raw[:, 0],
        "raw_prob_hgg": val_p_raw[:, 1],
        "cal_prob_lgg": val_p_cal[:, 0],
        "cal_prob_hgg": val_p_cal[:, 1],
    })
    val_csv_path = OUT_DIR / "val_logits_and_predictions.csv"
    df_val_export.to_csv(val_csv_path, index=False)
    print(f"    Saved Validation Logits: {val_csv_path.name}")

    # ---------------------------------------------------------------------------
    # Step 4: FREEZE T* and Verify Strict Data Isolation
    # ---------------------------------------------------------------------------
    FROZEN_T = optimal_T
    print(f"\n[4] Parameter T* = {FROZEN_T:.6f} is now FROZEN.")
    print("    Strict isolation guaranteed: Locked test set did NOT participate in fitting T*.")

    # ---------------------------------------------------------------------------
    # Step 5: Forward Pass on LOCKED TEST Split
    # ---------------------------------------------------------------------------
    print("\n[5] Loading Locked Test Cohort (Read-Only Evaluation)...")
    df_test_raw = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test_raw[df_test_raw["task"] == "GLI"].reset_index(drop=True)
    ds_test = BraTS2023Dataset(df_test, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    test_loader = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    n_test_lgg = int((df_test["label_grade"] == 0).sum())
    n_test_hgg = int((df_test["label_grade"] == 1).sum())
    print(f"    Test subjects: {len(df_test)} (LGG={n_test_lgg}, HGG={n_test_hgg})")

    test_sub_ids, test_y_true, test_logits = run_forward_pass(model, test_loader, device, desc="Test Forward Pass")
    print(f"    Test logits extracted. Shape: {test_logits.shape}")

    # ---------------------------------------------------------------------------
    # Step 6: Apply Frozen T* to Locked Test Logits
    # ---------------------------------------------------------------------------
    print("\n[6] Applying Frozen Temperature T* to Locked Test Logits...")
    test_nll_raw = compute_nll(test_logits, test_y_true, temp=1.0)
    test_nll_cal = compute_nll(test_logits, test_y_true, temp=FROZEN_T)

    test_p_raw = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
    test_p_cal = torch.softmax(torch.from_numpy(test_logits / FROZEN_T), dim=1).numpy()

    # Calculate test calibration metrics
    test_cal_raw = compute_calibration_data(test_y_true, test_p_raw[:, 1], test_p_raw[:, 0], n_bins=10)
    test_cal_opt = compute_calibration_data(test_y_true, test_p_cal[:, 1], test_p_cal[:, 0], n_bins=10)

    # Calculate standard threshold 0.50 classification metrics
    pred_raw_05 = (test_p_raw[:, 1] >= 0.5).astype(int)
    pred_cal_05 = (test_p_cal[:, 1] >= 0.5).astype(int)

    cm_raw = confusion_matrix(test_y_true, pred_raw_05, labels=[0, 1])
    cm_cal = confusion_matrix(test_y_true, pred_cal_05, labels=[0, 1])

    tn_r, fp_r, fn_r, tp_r = cm_raw.ravel()
    tn_c, fp_c, fn_c, tp_c = cm_cal.ravel()

    acc_raw = float(accuracy_score(test_y_true, pred_raw_05))
    acc_cal = float(accuracy_score(test_y_true, pred_cal_05))

    f1_m_raw = float(f1_score(test_y_true, pred_raw_05, average="macro"))
    f1_m_cal = float(f1_score(test_y_true, pred_cal_05, average="macro"))

    bal_acc_raw = float(balanced_accuracy_score(test_y_true, pred_raw_05))
    bal_acc_cal = float(balanced_accuracy_score(test_y_true, pred_cal_05))

    lgg_rec_raw = float(tn_r / (tn_r + fp_r))
    lgg_rec_cal = float(tn_c / (tn_c + fp_c))

    hgg_rec_raw = float(tp_r / (tp_r + fn_r))
    hgg_rec_cal = float(tp_c / (tp_c + fn_c))

    auc_raw = float(roc_auc_score(test_y_true, test_p_raw[:, 1]))
    auc_cal = float(roc_auc_score(test_y_true, test_p_cal[:, 1]))

    # Calculate uncertainty (entropy)
    p0c = np.clip(test_p_cal[:, 0], 1e-12, 1.0)
    p1c = np.clip(test_p_cal[:, 1], 1e-12, 1.0)
    cal_uncertainty = -(p0c * np.log2(p0c) + p1c * np.log2(p1c))
    cal_confidence = np.maximum(test_p_cal[:, 0], test_p_cal[:, 1])

    # Save per-subject test calibrated predictions
    df_test_export = pd.DataFrame({
        "subject_id": test_sub_ids,
        "true_label": test_y_true,
        "logit_lgg": test_logits[:, 0],
        "logit_hgg": test_logits[:, 1],
        "raw_prob_lgg": test_p_raw[:, 0],
        "raw_prob_hgg": test_p_raw[:, 1],
        "cal_prob_lgg": test_p_cal[:, 0],
        "cal_prob_hgg": test_p_cal[:, 1],
        "cal_predicted_label_05": pred_cal_05,
        "cal_confidence": cal_confidence,
        "cal_uncertainty": cal_uncertainty,
    })
    test_csv_path = OUT_DIR / "test_calibrated_predictions.csv"
    df_test_export.to_csv(test_csv_path, index=False)
    print(f"    Saved Test Calibrated Predictions: {test_csv_path.name}")

    # Save test calibration bins CSV
    df_hgg_bins_cal = pd.DataFrame(test_cal_opt["hgg_bins"])
    df_lgg_bins_cal = pd.DataFrame(test_cal_opt["lgg_bins"])
    df_all_bins_cal = pd.concat([df_hgg_bins_cal, df_lgg_bins_cal], ignore_index=True)
    bins_csv_path = OUT_DIR / "test_calibration_bins.csv"
    df_all_bins_cal.to_csv(bins_csv_path, index=False)
    print(f"    Saved Test Calibration Bins: {bins_csv_path.name}")

    # Save temperature parameter JSON
    param_json = {
        "frozen_temperature_T": round(FROZEN_T, 6),
        "optimization_dataset": "BraTS 2023 GLI Validation Split (n=187)",
        "optimization_loss": "Negative Log-Likelihood (CrossEntropyLoss)",
        "optimization_algorithm": "Scipy Bounded Scalar Minimization",
        "optimization_success": opt_res.success,
        "optimization_evaluations": opt_res.nfev,
        "validation_metrics": {
            "n_subjects": len(val_y_true),
            "lgg_subjects": n_val_lgg,
            "hgg_subjects": n_val_hgg,
            "initial_nll_T1": round(val_nll_initial, 6),
            "calibrated_nll_Topt": round(val_nll_calibrated, 6),
            "delta_nll": round(val_nll_calibrated - val_nll_initial, 6),
            "raw_confidence_ece": val_cal_raw["confidence_ece"],
            "calibrated_confidence_ece": val_cal_opt["confidence_ece"],
            "raw_brier_score": val_cal_raw["brier_score"],
            "calibrated_brier_score": val_cal_opt["brier_score"],
        },
    }
    param_json_path = OUT_DIR / "temperature_parameter.json"
    param_json_path.write_text(json.dumps(param_json, indent=2))
    print(f"    Saved Temperature Parameters: {param_json_path.name}")

    # Load CE baseline if available
    ce_metrics_path = RESULTS_DIR / "locked_test" / "protocol_A" / "resnet18_metrics.json"
    ce_metrics = json.loads(ce_metrics_path.read_text()) if ce_metrics_path.exists() else {}

    # Save test calibration metrics JSON
    metrics_summary_json = {
        "experiment": "Post-Hoc Temperature Scaling for Focal Loss ResNet18",
        "frozen_temperature_T": round(FROZEN_T, 6),
        "test_cohort": "BraTS 2023 GLI Locked Test (188 subjects: 12 LGG, 176 HGG)",
        "calibration_comparison": {
            "brier_score": {
                "focal_uncalibrated": test_cal_raw["brier_score"],
                "focal_temperature_scaled": test_cal_opt["brier_score"],
                "delta": round(test_cal_opt["brier_score"] - test_cal_raw["brier_score"], 4),
                "ce_baseline": 0.0248,
            },
            "confidence_ece": {
                "focal_uncalibrated": test_cal_raw["confidence_ece"],
                "focal_temperature_scaled": test_cal_opt["confidence_ece"],
                "delta": round(test_cal_opt["confidence_ece"] - test_cal_raw["confidence_ece"], 4),
                "ce_baseline": 0.0276,
            },
            "confidence_mce": {
                "focal_uncalibrated": test_cal_raw["confidence_mce"],
                "focal_temperature_scaled": test_cal_opt["confidence_mce"],
                "delta": round(test_cal_opt["confidence_mce"] - test_cal_raw["confidence_mce"], 4),
                "ce_baseline": 0.2864,
            },
            "hgg_prob_ece": {
                "focal_uncalibrated": test_cal_raw["hgg_prob_ece"],
                "focal_temperature_scaled": test_cal_opt["hgg_prob_ece"],
                "delta": round(test_cal_opt["hgg_prob_ece"] - test_cal_raw["hgg_prob_ece"], 4),
                "ce_baseline": 0.0369,
            },
            "hgg_prob_mce": {
                "focal_uncalibrated": test_cal_raw["hgg_prob_mce"],
                "focal_temperature_scaled": test_cal_opt["hgg_prob_mce"],
                "delta": round(test_cal_opt["hgg_prob_mce"] - test_cal_raw["hgg_prob_mce"], 4),
                "ce_baseline": 0.4472,
            },
            "lgg_prob_ece": {
                "focal_uncalibrated": test_cal_raw["lgg_prob_ece"],
                "focal_temperature_scaled": test_cal_opt["lgg_prob_ece"],
                "delta": round(test_cal_opt["lgg_prob_ece"] - test_cal_raw["lgg_prob_ece"], 4),
                "ce_baseline": 0.0369,
            },
        },
        "classification_comparison_at_threshold_050": {
            "accuracy": {"raw": acc_raw, "calibrated": acc_cal, "ce_baseline": ce_metrics.get("accuracy")},
            "macro_f1": {"raw": f1_m_raw, "calibrated": f1_m_cal, "ce_baseline": ce_metrics.get("f1_macro")},
            "balanced_accuracy": {"raw": bal_acc_raw, "calibrated": bal_acc_cal},
            "lgg_recall": {"raw": lgg_rec_raw, "calibrated": lgg_rec_cal, "ce_baseline": ce_metrics.get("lgg_recall")},
            "hgg_recall": {"raw": hgg_rec_raw, "calibrated": hgg_rec_cal, "ce_baseline": ce_metrics.get("hgg_recall")},
            "roc_auc": {"raw": auc_raw, "calibrated": auc_cal, "ce_baseline": ce_metrics.get("auc")},
            "confusion_matrix_raw": {"tn": int(tn_r), "fp": int(fp_r), "fn": int(fn_r), "tp": int(tp_r)},
            "confusion_matrix_calibrated": {"tn": int(tn_c), "fp": int(fp_c), "fn": int(fn_c), "tp": int(tp_c)},
        },
        "test_nll": {
            "raw_T1": round(test_nll_raw, 6),
            "calibrated_Topt": round(test_nll_cal, 6),
            "delta": round(test_nll_cal - test_nll_raw, 6),
        },
    }
    metrics_json_path = OUT_DIR / "test_calibration_metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_summary_json, indent=2))
    print(f"    Saved Test Calibration Metrics: {metrics_json_path.name}")

    # ---------------------------------------------------------------------------
    # Step 7: Generate Publication-Quality Figures
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Plot A: Validation Calibration Curve (Before vs After)
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration", linewidth=1.5)
    v_raw_x = [b["mean_predicted_prob"] for b in val_cal_raw["hgg_bins"]]
    v_raw_y = [b["observed_fraction"] for b in val_cal_raw["hgg_bins"]]
    v_cal_x = [b["mean_predicted_prob"] for b in val_cal_opt["hgg_bins"]]
    v_cal_y = [b["observed_fraction"] for b in val_cal_opt["hgg_bins"]]
    ax1.plot(v_raw_x, v_raw_y, marker="s", color="#d95f02", linewidth=1.8, label=f"Val Raw (T=1.0, ECE={val_cal_raw['confidence_ece']:.3f})")
    ax1.plot(v_cal_x, v_cal_y, marker="o", color="#1b9e77", linewidth=2.0, label=f"Val Calibrated (T={FROZEN_T:.2f}, ECE={val_cal_opt['confidence_ece']:.3f})")
    ax1.set_title(f"A. Validation Split Calibration (T*={FROZEN_T:.2f})", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Mean Predicted Probability P(HGG)", fontsize=10.5)
    ax1.set_ylabel("Observed Fraction of HGG", fontsize=10.5)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5)

    # Plot B: Locked-Test Reliability Diagram (Before vs After)
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration", linewidth=1.5)
    t_raw_x = [b["mean_predicted_prob"] for b in test_cal_raw["hgg_bins"]]
    t_raw_y = [b["observed_fraction"] for b in test_cal_raw["hgg_bins"]]
    t_cal_x = [b["mean_predicted_prob"] for b in test_cal_opt["hgg_bins"]]
    t_cal_y = [b["observed_fraction"] for b in test_cal_opt["hgg_bins"]]
    ax2.plot(t_raw_x, t_raw_y, marker="s", color="#d95f02", linestyle="--", linewidth=1.8, label=f"Test Raw (ECE={test_cal_raw['confidence_ece']:.3f})")
    ax2.plot(t_cal_x, t_cal_y, marker="o", color="#2b5c8f", linewidth=2.2, label=f"Test Temp-Scaled (ECE={test_cal_opt['confidence_ece']:.3f})")
    ax2.set_title("B. Locked-Test Reliability Diagram", fontsize=11.5, fontweight="bold")
    ax2.set_xlabel("Mean Predicted Probability P(HGG)", fontsize=10.5)
    ax2.set_ylabel("Observed Fraction of HGG", fontsize=10.5)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=8.5)

    # Plot C: Test Probability Distribution (Raw vs Calibrated)
    ax3 = axes[2]
    ax3.hist(test_p_raw[:, 1], bins=20, range=(0, 1), alpha=0.5, color="#d95f02", label="Raw P(HGG) (T=1.0)", edgecolor="black")
    ax3.hist(test_p_cal[:, 1], bins=20, range=(0, 1), alpha=0.5, color="#2b5c8f", label=f"Scaled P(HGG) (T={FROZEN_T:.2f})", edgecolor="black")
    ax3.axvline(0.50, color="red", linestyle="--", label="Threshold = 0.50")
    ax3.set_title("C. Test P(HGG) Density Shift", fontsize=11.5, fontweight="bold")
    ax3.set_xlabel("Predicted Probability P(HGG)", fontsize=10.5)
    ax3.set_ylabel("Subject Count", fontsize=10.5)
    ax3.set_xlim(-0.02, 1.02)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plot_path = OUT_DIR / "focal_temperature_scaling_diagram.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"    Saved Plot: {plot_path.name}")

    # ---------------------------------------------------------------------------
    # Step 8: Console Summary Report
    # ---------------------------------------------------------------------------
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("EXPERIMENT EXECUTION COMPLETE")
    print(f"Total Runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
    print("=" * 80)
    print(f"Frozen Optimal Temperature: T* = {FROZEN_T:.4f}")
    print(f"Validation NLL:             {val_nll_initial:.4f} -> {val_nll_calibrated:.4f} (Delta: {val_nll_calibrated - val_nll_initial:+.4f})")
    print(f"Validation ECE:             {val_cal_raw['confidence_ece']:.4f} -> {val_cal_opt['confidence_ece']:.4f}")
    print(f"\nLocked Test Calibration:")
    print(f"  Test NLL:                 {test_nll_raw:.4f} -> {test_nll_cal:.4f} (Delta: {test_nll_cal - test_nll_raw:+.4f})")
    print(f"  Test Confidence ECE:      {test_cal_raw['confidence_ece']:.4f} -> {test_cal_opt['confidence_ece']:.4f} (Delta: {test_cal_opt['confidence_ece'] - test_cal_raw['confidence_ece']:+.4f})")
    print(f"  Test Brier Score:         {test_cal_raw['brier_score']:.4f} -> {test_cal_opt['brier_score']:.4f} (Delta: {test_cal_opt['brier_score'] - test_cal_raw['brier_score']:+.4f})")
    print(f"  Test HGG Prob ECE:        {test_cal_raw['hgg_prob_ece']:.4f} -> {test_cal_opt['hgg_prob_ece']:.4f} (Delta: {test_cal_opt['hgg_prob_ece'] - test_cal_raw['hgg_prob_ece']:+.4f})")
    print(f"\nLocked Test Classification (Threshold = 0.50):")
    print(f"  Accuracy:                 {acc_raw:.4f} -> {acc_cal:.4f}")
    print(f"  Macro-F1:                 {f1_m_raw:.4f} -> {f1_m_cal:.4f}")
    print(f"  LGG Recall:               {lgg_rec_raw:.4f} ({tn_r}/{tn_r+fp_r}) -> {lgg_rec_cal:.4f} ({tn_c}/{tn_c+fp_c})")
    print(f"  HGG Recall:               {hgg_rec_raw:.4f} ({tp_r}/{tp_r+fn_r}) -> {hgg_rec_cal:.4f} ({tp_c}/{tp_c+fn_c})")
    print(f"  ROC-AUC:                  {auc_raw:.4f} -> {auc_cal:.4f} (Strict invariance: {auc_raw == auc_cal})")
    print("=" * 80)


if __name__ == "__main__":
    main()
