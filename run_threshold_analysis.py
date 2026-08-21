"""
run_threshold_analysis.py
-------------------------
Day 6 — Final Task: Decision-Threshold Sensitivity & Operating-Point Analysis.

Safety & Freeze Rules:
- STRICT READ-ONLY inference on development validation set
- NEVER tune threshold directly on locked test set
- Frozen locked-test probabilities used for out-of-sample evaluation ONLY
- Checkpoint and manifest integrity preserved 100%
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    RESULTS_DIR,
    BRATS2023_VAL_MANIFEST,
    BRATS2023_TEST_MANIFEST,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model, get_training_device

ARCHIVE_DIR = RESULTS_DIR / "protocol_archives"
LOCKED_TEST_DIR = RESULTS_DIR / "locked_test"
OUT_DIR = RESULTS_DIR / "reliability_analysis" / "threshold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOLS = ["protocol_A", "protocol_B", "protocol_C"]
MODELS = {
    "resnet18_medicalnet": "resnet18_medicalnet_lgg_hgg_3d.pt",
    "densenet121_3d":      "densenet121_3d_lgg_hgg_3d.pt",
    "swin_transformer_3d": "swin_transformer_3d_lgg_hgg_3d.pt",
}
MODEL_DISPLAY_NAMES = {
    "resnet18_medicalnet": "ResNet18 MedicalNet",
    "densenet121_3d":      "DenseNet121 3D",
    "swin_transformer_3d": "SwinUNETR 3D",
}
FILE_PREFIX_MAP = {
    "resnet18_medicalnet": "resnet18",
    "densenet121_3d":      "densenet121",
    "swin_transformer_3d": "swin",
}


def compute_metrics_at_threshold(y_true: np.ndarray, prob_lgg: np.ndarray, tau_lgg: float) -> Dict:
    """
    Decision rule:
      y_pred = 0 (LGG) if prob_lgg >= tau_lgg else 1 (HGG)
    """
    y_pred = (prob_lgg < tau_lgg).astype(int)  # 1 if prob_lgg < tau, else 0

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    lgg_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Sensitivity for class 0
    hgg_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Specificity for class 1
    precision_lgg = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    precision_hgg = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    pred_lgg_prop = float((y_pred == 0).mean())

    return {
        "threshold_tau_lgg": round(tau_lgg, 4),
        "threshold_theta_hgg": round(1.0 - tau_lgg, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "lgg_recall": round(lgg_recall, 4),
        "hgg_recall": round(hgg_recall, 4),
        "precision_lgg": round(precision_lgg, 4),
        "precision_hgg": round(precision_hgg, 4),
        "pred_lgg_prop": round(pred_lgg_prop, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    print("=" * 80)
    print("DAY 6 — FINAL TASK: THRESHOLD ANALYSIS & OPERATING-POINT SENSITIVITY")
    print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    device = get_training_device()
    print(f"Device: {device}")

    # 1. Load Validation Dataset (187 subjects: 10 LGG, 177 HGG)
    df_val_raw = load_manifest(BRATS2023_VAL_MANIFEST, label_col="label_grade")
    df_val = df_val_raw[df_val_raw["task"] == "GLI"].reset_index(drop=True)
    val_ds = BraTS2023Dataset(df_val, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Validation cohort: {len(df_val)} subjects (LGG={(df_val['label_grade']==0).sum()}, HGG={(df_val['label_grade']==1).sum()})")

    # Cache pre-processed validation batches in RAM for ultra-fast multi-model forward passes
    print("Pre-caching validation tensors in memory...")
    t0_cache = time.time()
    val_batches = []
    for batch in val_loader:
        val_batches.append({
            "image": batch["image"],
            "label": batch["label"],
            "subject_id": batch.get("subject_id", [])
        })
    print(f"Cached {len(val_batches)} validation batches in {time.time()-t0_cache:.2f}s")

    # 2. Forward inference across 9 canonical models on validation set
    val_predictions = {}
    for proto in PROTOCOLS:
        proto_short = proto.split("_")[1]
        for mkey, mfile in MODELS.items():
            disp_name = MODEL_DISPLAY_NAMES[mkey]
            cfg_id = f"{proto_short}_{mkey}"
            ckpt_path = ARCHIVE_DIR / proto / mfile

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = get_model(mkey, num_classes=2)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)
            model.eval()

            y_true, y_prob_0, y_prob_1 = [], [], []
            with torch.no_grad():
                for batch in val_batches:
                    imgs = batch["image"].to(device)
                    labels = batch["label"]
                    with torch.amp.autocast(device_type=device.type, enabled=True):
                        logits = model(imgs)
                        probs = torch.softmax(logits, dim=1)
                    y_true.extend(labels.numpy())
                    probs_np = probs.cpu().numpy()
                    y_prob_0.extend(probs_np[:, 0])
                    y_prob_1.extend(probs_np[:, 1])

            y_true = np.array(y_true)
            y_prob_0 = np.array(y_prob_0)
            y_prob_1 = np.array(y_prob_1)
            val_auc = float(roc_auc_score(y_true, y_prob_1))

            val_predictions[cfg_id] = {
                "protocol": proto_short,
                "model_key": mkey,
                "model_display_name": disp_name,
                "y_true": y_true,
                "prob_lgg": y_prob_0,
                "prob_hgg": y_prob_1,
                "auc": val_auc,
            }
            print(f"  [Val Eval] {proto_short} | {disp_name:<20}: AUC = {val_auc:.4f}")
            del model
            torch.cuda.empty_cache()

    # 3. Load Frozen Locked-Test Predictions (188 subjects: 12 LGG, 176 HGG)
    test_predictions = {}
    for proto in PROTOCOLS:
        proto_short = proto.split("_")[1]
        for mkey in MODELS.keys():
            prefix = FILE_PREFIX_MAP[mkey]
            cfg_id = f"{proto_short}_{mkey}"
            csv_path = LOCKED_TEST_DIR / proto / f"{prefix}_predictions.csv"
            df_test = pd.read_csv(csv_path)

            y_true_test = df_test["true_label"].values
            prob_lgg_test = df_test["prob_lgg"].values
            prob_hgg_test = df_test["prob_hgg"].values
            test_auc = float(roc_auc_score(y_true_test, prob_hgg_test))

            test_predictions[cfg_id] = {
                "protocol": proto_short,
                "model_key": mkey,
                "model_display_name": MODEL_DISPLAY_NAMES[mkey],
                "y_true": y_true_test,
                "prob_lgg": prob_lgg_test,
                "prob_hgg": prob_hgg_test,
                "auc": test_auc,
            }

    # 4. Systematic Threshold Sweep on Validation Data
    threshold_grid = np.round(np.linspace(0.01, 0.99, 99), 4)

    sweep_records = []
    selected_rules_records = []
    comparison_records = []

    for cfg_id, val_data in val_predictions.items():
        proto = val_data["protocol"]
        disp_name = val_data["model_display_name"]
        mkey = val_data["model_key"]
        y_val = val_data["y_true"]
        p_lgg_val = val_data["prob_lgg"]
        val_auc = val_data["auc"]

        test_data = test_predictions[cfg_id]
        y_test = test_data["y_true"]
        p_lgg_test = test_data["prob_lgg"]
        test_auc = test_data["auc"]

        # Run sweep across all thresholds on validation set
        val_sweep_results = []
        for tau in threshold_grid:
            m_dict = compute_metrics_at_threshold(y_val, p_lgg_val, tau)
            m_dict["protocol"] = proto
            m_dict["model_name"] = disp_name
            m_dict["model_key"] = mkey
            m_dict["split"] = "validation"
            m_dict["auc"] = val_auc
            sweep_records.append(m_dict)
            val_sweep_results.append(m_dict)

        df_v_sweep = pd.DataFrame(val_sweep_results)

        # -------------------------------------------------------------
        # Evaluate Candidate Threshold Selection Rules on Validation Data
        # -------------------------------------------------------------

        # Rule D: Default 0.50 Threshold
        tau_default = 0.50
        val_m_default = compute_metrics_at_threshold(y_val, p_lgg_val, tau_default)
        test_m_default = compute_metrics_at_threshold(y_test, p_lgg_test, tau_default)

        # Rule A: Maximize Validation Macro-F1
        best_f1_val = df_v_sweep["macro_f1"].max()
        f1_tied = df_v_sweep[df_v_sweep["macro_f1"] >= best_f1_val - 1e-4]
        # Tie breaker: highest LGG recall, then closest to 0.50
        best_f1_row = f1_tied.sort_values(by=["lgg_recall", "balanced_accuracy"], ascending=[False, False]).iloc[0]
        tau_rule_A = float(best_f1_row["threshold_tau_lgg"])
        val_m_rule_A = compute_metrics_at_threshold(y_val, p_lgg_val, tau_rule_A)
        test_m_rule_A = compute_metrics_at_threshold(y_test, p_lgg_test, tau_rule_A)

        # Rule B: Maximize LGG Recall subject to Specificity (HGG Recall) >= 0.90
        spec_90_candidates = df_v_sweep[df_v_sweep["hgg_recall"] >= 0.90]
        if len(spec_90_candidates) > 0:
            max_rec_90 = spec_90_candidates["lgg_recall"].max()
            rule_B_row = spec_90_candidates[spec_90_candidates["lgg_recall"] == max_rec_90].sort_values(by=["macro_f1"], ascending=False).iloc[0]
            tau_rule_B = float(rule_B_row["threshold_tau_lgg"])
        else:
            tau_rule_B = tau_default
        val_m_rule_B = compute_metrics_at_threshold(y_val, p_lgg_val, tau_rule_B)
        test_m_rule_B = compute_metrics_at_threshold(y_test, p_lgg_test, tau_rule_B)

        # Rule C: Maximize Balanced Accuracy (Youden's J Index)
        best_bal_acc = df_v_sweep["balanced_accuracy"].max()
        bal_tied = df_v_sweep[df_v_sweep["balanced_accuracy"] >= best_bal_acc - 1e-4]
        rule_C_row = bal_tied.sort_values(by=["macro_f1"], ascending=False).iloc[0]
        tau_rule_C = float(rule_C_row["threshold_tau_lgg"])
        val_m_rule_C = compute_metrics_at_threshold(y_val, p_lgg_val, tau_rule_C)
        test_m_rule_C = compute_metrics_at_threshold(y_test, p_lgg_test, tau_rule_C)

        # Store Selected Rules Records
        selected_rules_records.append({
            "Protocol": proto,
            "Model": disp_name,
            "model_key": mkey,
            "Val_AUC": round(val_auc, 4),
            "Test_AUC": round(test_auc, 4),
            "Tau_Default": tau_default,
            "Val_F1_Default": val_m_default["macro_f1"],
            "Val_LGG_Rec_Default": val_m_default["lgg_recall"],
            "Test_F1_Default": test_m_default["macro_f1"],
            "Test_LGG_Rec_Default": test_m_default["lgg_recall"],
            "Test_HGG_Rec_Default": test_m_default["hgg_recall"],
            
            "Tau_Rule_A_MaxF1": tau_rule_A,
            "Val_F1_Rule_A": val_m_rule_A["macro_f1"],
            "Val_LGG_Rec_Rule_A": val_m_rule_A["lgg_recall"],
            "Test_F1_Rule_A": test_m_rule_A["macro_f1"],
            "Test_LGG_Rec_Rule_A": test_m_rule_A["lgg_recall"],
            "Test_HGG_Rec_Rule_A": test_m_rule_A["hgg_recall"],

            "Tau_Rule_B_Spec90": tau_rule_B,
            "Val_F1_Rule_B": val_m_rule_B["macro_f1"],
            "Val_LGG_Rec_Rule_B": val_m_rule_B["lgg_recall"],
            "Test_F1_Rule_B": test_m_rule_B["macro_f1"],
            "Test_LGG_Rec_Rule_B": test_m_rule_B["lgg_recall"],
            "Test_HGG_Rec_Rule_B": test_m_rule_B["hgg_recall"],

            "Tau_Rule_C_BalAcc": tau_rule_C,
            "Val_F1_Rule_C": val_m_rule_C["macro_f1"],
            "Val_LGG_Rec_Rule_C": val_m_rule_C["lgg_recall"],
            "Test_F1_Rule_C": test_m_rule_C["macro_f1"],
            "Test_LGG_Rec_Rule_C": test_m_rule_C["lgg_recall"],
            "Test_HGG_Rec_Rule_C": test_m_rule_C["hgg_recall"],
        })

        # Detailed Comparison Record (Default vs Rule A Optimal)
        comparison_records.append({
            "Protocol": proto,
            "Model": disp_name,
            "model_key": mkey,
            "Default_Tau": tau_default,
            "Selected_Tau": tau_rule_A,
            "Test_Acc_Default": test_m_default["accuracy"],
            "Test_Acc_Selected": test_m_rule_A["accuracy"],
            "Test_F1_Default": test_m_default["macro_f1"],
            "Test_F1_Selected": test_m_rule_A["macro_f1"],
            "Test_LGG_Rec_Default": test_m_default["lgg_recall"],
            "Test_LGG_Rec_Selected": test_m_rule_A["lgg_recall"],
            "Test_HGG_Rec_Default": test_m_default["hgg_recall"],
            "Test_HGG_Rec_Selected": test_m_rule_A["hgg_recall"],
            "Test_AUC": round(test_auc, 4),
            "LGG_Recall_Delta": round(test_m_rule_A["lgg_recall"] - test_m_default["lgg_recall"], 4),
            "HGG_Recall_Delta": round(test_m_rule_A["hgg_recall"] - test_m_default["hgg_recall"], 4),
            "Macro_F1_Delta": round(test_m_rule_A["macro_f1"] - test_m_default["macro_f1"], 4),
        })

    # Save CSV tables
    df_all_sweeps = pd.DataFrame(sweep_records)
    df_all_sweeps.to_csv(OUT_DIR / "threshold_sweep_results.csv", index=False)

    df_selected = pd.DataFrame(selected_rules_records)
    df_selected.to_csv(OUT_DIR / "threshold_selected_results.csv", index=False)

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(OUT_DIR / "threshold_comparison.csv", index=False)

    print("\nSaved CSV tables:")
    print("  - threshold_sweep_results.csv")
    print("  - threshold_selected_results.csv")
    print("  - threshold_comparison.csv")

    # 5. Generate Visualizations & Plots
    print("\nGenerating threshold sensitivity plots...")

    # Plot 1: Macro-F1 vs Threshold (Protocol A Models)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"resnet18_medicalnet": "#1f77b4", "densenet121_3d": "#2ca02c", "swin_transformer_3d": "#d62728"}
    for mkey, disp_name in MODEL_DISPLAY_NAMES.items():
        sub_df = df_all_sweeps[(df_all_sweeps["protocol"] == "A") & (df_all_sweeps["model_key"] == mkey)]
        ax.plot(sub_df["threshold_tau_lgg"], sub_df["macro_f1"], label=f"{disp_name}", linewidth=2.2, color=colors[mkey])
    ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax.set_xlabel("LGG Probability Threshold ($\tau_{LGG}$)")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("Decision-Threshold Sensitivity — Validation Macro-F1 (Protocol A)")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0.3, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "macro_f1_vs_threshold.png", dpi=300)
    plt.close()

    # Plot 2: LGG Recall vs Threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    for mkey, disp_name in MODEL_DISPLAY_NAMES.items():
        sub_df = df_all_sweeps[(df_all_sweeps["protocol"] == "A") & (df_all_sweeps["model_key"] == mkey)]
        ax.plot(sub_df["threshold_tau_lgg"], sub_df["lgg_recall"], label=f"{disp_name}", linewidth=2.2, color=colors[mkey])
    ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax.set_xlabel("LGG Probability Threshold ($\tau_{LGG}$)")
    ax.set_ylabel("Validation LGG Recall (Sensitivity)")
    ax.set_title("Decision-Threshold Sensitivity — LGG Recall (Protocol A)")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lgg_recall_vs_threshold.png", dpi=300)
    plt.close()

    # Plot 3: Specificity (HGG Recall) vs Threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    for mkey, disp_name in MODEL_DISPLAY_NAMES.items():
        sub_df = df_all_sweeps[(df_all_sweeps["protocol"] == "A") & (df_all_sweeps["model_key"] == mkey)]
        ax.plot(sub_df["threshold_tau_lgg"], sub_df["hgg_recall"], label=f"{disp_name}", linewidth=2.2, color=colors[mkey])
    ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax.set_xlabel("LGG Probability Threshold ($\tau_{LGG}$)")
    ax.set_ylabel("Validation HGG Specificity (HGG Recall)")
    ax.set_title("Decision-Threshold Sensitivity — HGG Specificity (Protocol A)")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0.4, 1.02])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "specificity_vs_threshold.png", dpi=300)
    plt.close()

    # Plot 4: Balanced Accuracy vs Threshold
    fig, ax = plt.subplots(figsize=(8, 5))
    for mkey, disp_name in MODEL_DISPLAY_NAMES.items():
        sub_df = df_all_sweeps[(df_all_sweeps["protocol"] == "A") & (df_all_sweeps["model_key"] == mkey)]
        ax.plot(sub_df["threshold_tau_lgg"], sub_df["balanced_accuracy"], label=f"{disp_name}", linewidth=2.2, color=colors[mkey])
    ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.50)")
    ax.set_xlabel("LGG Probability Threshold ($\tau_{LGG}$)")
    ax.set_ylabel("Validation Balanced Accuracy")
    ax.set_title("Decision-Threshold Sensitivity — Balanced Accuracy (Protocol A)")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0.4, 1.0])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "balanced_accuracy_vs_threshold.png", dpi=300)
    plt.close()

    # Plot 5: Tradeoff Curves (LGG Sensitivity vs HGG Specificity)
    fig, ax = plt.subplots(figsize=(8, 6))
    for mkey, disp_name in MODEL_DISPLAY_NAMES.items():
        sub_df = df_all_sweeps[(df_all_sweeps["protocol"] == "A") & (df_all_sweeps["model_key"] == mkey)]
        ax.plot(1.0 - sub_df["hgg_recall"], sub_df["lgg_recall"], label=f"{disp_name} (AUC={val_predictions['A_'+mkey]['auc']:.4f})", linewidth=2.5, color=colors[mkey])
        
        # Mark Default 0.50 operating point
        row_def = sub_df[sub_df["threshold_tau_lgg"] == 0.50].iloc[0]
        ax.scatter([1.0 - row_def["hgg_recall"]], [row_def["lgg_recall"]], color=colors[mkey], s=90, zorder=5, edgecolors="black")
    
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Chance")
    ax.set_xlabel("1 - HGG Specificity (False LGG Alarm Rate)")
    ax.set_ylabel("LGG Sensitivity (LGG Recall)")
    ax.set_title("Empirical ROC Operating Curves — Validation Cohort (Protocol A)")
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_operating_points.png", dpi=300)
    plt.close()

    print("Saved 5 threshold sensitivity plots successfully.")


if __name__ == "__main__":
    main()
