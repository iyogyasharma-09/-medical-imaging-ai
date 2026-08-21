"""
analyze_focal_thresholds.py
----------------------------
Threshold sensitivity & operating-point analysis for Focal Loss ResNet18 MedicalNet
locked-test predictions.

Safety & Freeze Rules:
- Post-hoc analysis of existing predictions only (no training, no forward passes).
- Reads ONLY from results/focal_locked_test/resnet18_focal_test_predictions.csv.
- Saves all artifacts in an isolated directory: results/focal_locked_test/threshold_analysis/.
- Zero modifications to CE baseline files or existing threshold analysis files.
"""

import json
import numpy as np
import pandas as pd
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

BASE_DIR = Path(r"C:\mri\mri-")
INPUT_PRED_CSV = BASE_DIR / "results" / "focal_locked_test" / "resnet18_focal_test_predictions.csv"
OUT_DIR = BASE_DIR / "results" / "focal_locked_test" / "threshold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics_at_threshold(y_true: np.ndarray, prob_lgg: np.ndarray, tau_lgg: float) -> dict:
    """
    Decision rule based on LGG probability threshold tau_lgg:
      y_pred = 0 (LGG) if prob_lgg >= tau_lgg else 1 (HGG)
    Equivalently with HGG threshold theta_hgg = 1.0 - tau_lgg:
      y_pred = 1 (HGG) if prob_hgg >= theta_hgg else 0 (LGG)
    """
    y_pred = (prob_lgg < tau_lgg).astype(int)  # 1 (HGG) if prob_lgg < tau, else 0 (LGG)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # Class 0 = LGG, Class 1 = HGG
    lgg_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # LGG Sensitivity / True Negative Rate
    hgg_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # HGG Sensitivity / True Positive Rate
    lgg_precision = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    hgg_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Sensitivity/Specificity designations:
    # If LGG is primary target class (class 0):
    #   Sensitivity = LGG Recall, Specificity = HGG Recall
    # If HGG is primary target class (class 1):
    #   Sensitivity = HGG Recall, Specificity = LGG Recall
    return {
        "threshold_tau_lgg": round(tau_lgg, 4),
        "threshold_theta_hgg": round(1.0 - tau_lgg, 4),
        "lgg_recall": round(lgg_recall, 4),
        "hgg_recall": round(hgg_recall, 4),
        "lgg_precision": round(lgg_precision, 4),
        "hgg_precision": round(hgg_precision, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "sensitivity_lgg": round(lgg_recall, 4),
        "specificity_hgg": round(hgg_recall, 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main():
    print("=" * 80)
    print("FOCAL LOSS LOCKED-TEST THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    # 1. Load predictions
    if not INPUT_PRED_CSV.exists():
        raise FileNotFoundError(f"Input predictions file missing: {INPUT_PRED_CSV}")

    df = pd.read_csv(INPUT_PRED_CSV)
    print(f"Loaded predictions: {len(df)} subjects from {INPUT_PRED_CSV.name}")

    y_true = df["true_label"].values
    p_lgg = df["prob_lgg"].values
    p_hgg = df["prob_hgg"].values

    n_lgg = int((y_true == 0).sum())
    n_hgg = int((y_true == 1).sum())
    print(f"Cohort composition: Total={len(y_true)}, LGG (0)={n_lgg}, HGG (1)={n_hgg}")

    # Verify ROC-AUC invariance
    roc_auc = float(roc_auc_score(y_true, p_hgg))
    print(f"ROC-AUC (constant across all thresholds): {roc_auc:.4f}")

    # 2. Sweep thresholds from 0.10 to 0.90 in increments of 0.05
    thresholds = np.arange(0.10, 0.9001, 0.05)
    records = []
    for tau in thresholds:
        m = compute_metrics_at_threshold(y_true, p_lgg, float(tau))
        records.append(m)

    df_metrics = pd.DataFrame(records)

    # Reorder columns for clean CSV export
    csv_cols = [
        "threshold_tau_lgg",
        "threshold_theta_hgg",
        "lgg_recall",
        "hgg_recall",
        "lgg_precision",
        "hgg_precision",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "sensitivity_lgg",
        "specificity_hgg",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    df_metrics_export = df_metrics[csv_cols]

    csv_path = OUT_DIR / "focal_threshold_metrics.csv"
    df_metrics_export.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # 3. Identify Key Operating Points
    # 3.1. Original threshold = 0.50
    row_050 = df_metrics[np.isclose(df_metrics["threshold_tau_lgg"], 0.50)].iloc[0].to_dict()

    # 3.2. Maximum Macro-F1
    best_f1_val = df_metrics["macro_f1"].max()
    row_max_f1 = df_metrics[df_metrics["macro_f1"] == best_f1_val].sort_values(
        by=["lgg_recall", "balanced_accuracy"], ascending=[False, False]
    ).iloc[0].to_dict()

    # 3.3. Maximum Balanced Accuracy
    best_bal_acc_val = df_metrics["balanced_accuracy"].max()
    row_max_bal_acc = df_metrics[df_metrics["balanced_accuracy"] == best_bal_acc_val].sort_values(
        by=["macro_f1"], ascending=False
    ).iloc[0].to_dict()

    # Helper for closest recall
    def find_closest_lgg_recall(target_recall):
        diffs = np.abs(df_metrics["lgg_recall"] - target_recall)
        min_diff = diffs.min()
        closest_rows = df_metrics[diffs == min_diff]
        # Prefer higher HGG recall among ties
        best_match = closest_rows.sort_values(by=["hgg_recall", "macro_f1"], ascending=[False, False]).iloc[0]
        return best_match.to_dict(), float(min_diff)

    row_close_50, diff_50 = find_closest_lgg_recall(0.50)
    row_close_60, diff_60 = find_closest_lgg_recall(0.60)
    row_close_70, diff_70 = find_closest_lgg_recall(0.70)
    row_close_80, diff_80 = find_closest_lgg_recall(0.80)

    # Detailed Analysis JSON
    analysis_json_data = {
        "experiment": "ResNet18 MedicalNet + Focal Loss (alpha=1.0, gamma=2.0)",
        "cohort": "BraTS 2023 GLI Locked Test (188 subjects: 12 LGG, 176 HGG)",
        "roc_auc": round(roc_auc, 4),
        "roc_auc_note": "Invariance confirmed: ROC-AUC is rank-based and unaffected by decision threshold.",
        "operating_points": {
            "default_0_50": row_050,
            "max_macro_f1": row_max_f1,
            "max_balanced_accuracy": row_max_bal_acc,
            "closest_lgg_recall_50": {
                "target": 0.50,
                "absolute_difference": round(diff_50, 4),
                "metrics": row_close_50,
            },
            "closest_lgg_recall_60": {
                "target": 0.60,
                "absolute_difference": round(diff_60, 4),
                "metrics": row_close_60,
            },
            "closest_lgg_recall_70": {
                "target": 0.70,
                "absolute_difference": round(diff_70, 4),
                "metrics": row_close_70,
            },
            "closest_lgg_recall_80": {
                "target": 0.80,
                "absolute_difference": round(diff_80, 4),
                "metrics": row_close_80,
            },
        },
        "all_thresholds": records,
        "scientific_interpretation": {
            "tradeoff_observed": True,
            "summary": (
                "Lowering the LGG classification threshold (tau_lgg from 0.50 down to 0.40-0.35) "
                "substantially increases LGG recall (from 41.67% up to 66.67%-83.33%) while causing only "
                "a modest reduction in HGG recall (from 97.16% down to 94.89%-90.91%). "
                "This confirms an empirical threshold-dependent operating-point trade-off without asserting "
                "a full causal resolution of the minority class learning dynamics."
            ),
        },
    }

    json_path = OUT_DIR / "focal_threshold_analysis.json"
    json_path.write_text(json.dumps(analysis_json_data, indent=2))
    print(f"Saved JSON: {json_path}")

    # 4. Generate Publication-Quality Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Plot 1: LGG Recall vs HGG Recall across Thresholds
    ax1 = axes[0]
    taus = df_metrics["threshold_tau_lgg"].values
    lgg_recs = df_metrics["lgg_recall"].values * 100
    hgg_recs = df_metrics["hgg_recall"].values * 100
    macro_f1s = df_metrics["macro_f1"].values * 100
    bal_accs = df_metrics["balanced_accuracy"].values * 100

    ax1.plot(taus, lgg_recs, marker="o", color="#2b5c8f", linewidth=2.2, label="LGG Recall (%)")
    ax1.plot(taus, hgg_recs, marker="s", color="#d95f02", linewidth=2.2, label="HGG Recall (%)")
    ax1.plot(taus, macro_f1s, marker="^", color="#7570b3", linewidth=1.8, linestyle="--", label="Macro-F1 (×100)")
    ax1.plot(taus, bal_accs, marker="d", color="#1b9e77", linewidth=1.8, linestyle=":", label="Balanced Acc (×100)")

    # Highlight default 0.50 and optimal thresholds
    ax1.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default threshold (0.50)")
    ax1.axvline(row_max_f1["threshold_tau_lgg"], color="#7570b3", linestyle=":", alpha=0.8,
                label=f"Max Macro-F1 ({row_max_f1['threshold_tau_lgg']:.2f})")

    ax1.set_title("Focal ResNet18: Metric Trade-offs vs Decision Threshold ($\tau_{LGG}$)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Decision Threshold $\tau_{LGG}$ (Predict LGG if $P(LGG) \geq \tau$)", fontsize=11)
    ax1.set_ylabel("Metric Value (%)", fontsize=11)
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=9.5, framealpha=0.95)

    # Plot 2: Operating Characteristic Curve (LGG Recall vs HGG Specificity/Recall)
    ax2 = axes[1]
    # Trade-off curve: HGG Recall vs LGG Recall
    ax2.plot(lgg_recs, hgg_recs, marker="o", color="#800026", linewidth=2.2, zorder=3)

    # Annotate specific key thresholds
    key_points = [
        (0.50, "Default (0.50)"),
        (0.45, "tau=0.45"),
        (0.40, "tau=0.40 (Max F1)"),
        (0.35, "tau=0.35"),
        (0.30, "tau=0.30"),
        (0.25, "tau=0.25 (Max BalAcc)"),
    ]
    for tau_val, label_text in key_points:
        row_pt = df_metrics[np.isclose(df_metrics["threshold_tau_lgg"], tau_val)].iloc[0]
        x = row_pt["lgg_recall"] * 100
        y = row_pt["hgg_recall"] * 100
        ax2.scatter(x, y, s=80, color="#fc4e2a", edgecolor="black", zorder=4)
        ax2.annotate(
            f"{label_text}\n({x:.1f}%, {y:.1f}%)",
            (x, y),
            textcoords="offset points",
            xytext=(10, -10 if y > 90 else 10),
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.3, ec="gray"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black"),
        )

    ax2.set_title("Operating Characteristic: LGG Recall vs HGG Recall", fontsize=13, fontweight="bold")
    ax2.set_xlabel("LGG Recall / Minority Sensitivity (%)", fontsize=11)
    ax2.set_ylabel("HGG Recall / Majority Sensitivity (%)", fontsize=11)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(50, 105)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    png_path = OUT_DIR / "focal_threshold_analysis.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved Plot: {png_path}")

    # Print summary tables to console
    print("\n" + "=" * 105)
    print("FOCAL RESNET18 THRESHOLD SWEEP TABLE")
    print("=" * 105)
    print(df_metrics_export.to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY OPERATING POINTS SUMMARY")
    print("=" * 80)
    print(f"1. Default Threshold (tau=0.50):")
    print(f"   LGG Recall={row_050['lgg_recall']:.4f} ({row_050['tn']}/{row_050['tn']+row_050['fp']}), "
          f"HGG Recall={row_050['hgg_recall']:.4f} ({row_050['tp']}/{row_050['tp']+row_050['fn']}), "
          f"Macro-F1={row_050['macro_f1']:.4f}, BalAcc={row_050['balanced_accuracy']:.4f}")
    print(f"   Confusion Matrix: TN={row_050['tn']}, FP={row_050['fp']}, FN={row_050['fn']}, TP={row_050['tp']}")

    print(f"\n2. Maximum Macro-F1 Threshold (tau={row_max_f1['threshold_tau_lgg']:.2f}):")
    print(f"   Macro-F1={row_max_f1['macro_f1']:.4f}, LGG Recall={row_max_f1['lgg_recall']:.4f} ({row_max_f1['tn']}/12), "
          f"HGG Recall={row_max_f1['hgg_recall']:.4f} ({row_max_f1['tp']}/176), BalAcc={row_max_f1['balanced_accuracy']:.4f}")
    print(f"   Confusion Matrix: TN={row_max_f1['tn']}, FP={row_max_f1['fp']}, FN={row_max_f1['fn']}, TP={row_max_f1['tp']}")

    print(f"\n3. Maximum Balanced Accuracy Threshold (tau={row_max_bal_acc['threshold_tau_lgg']:.2f}):")
    print(f"   BalAcc={row_max_bal_acc['balanced_accuracy']:.4f}, LGG Recall={row_max_bal_acc['lgg_recall']:.4f} ({row_max_bal_acc['tn']}/12), "
          f"HGG Recall={row_max_bal_acc['hgg_recall']:.4f} ({row_max_bal_acc['tp']}/176), Macro-F1={row_max_bal_acc['macro_f1']:.4f}")
    print(f"   Confusion Matrix: TN={row_max_bal_acc['tn']}, FP={row_max_bal_acc['fp']}, FN={row_max_bal_acc['fn']}, TP={row_max_bal_acc['tp']}")

    print(f"\n4. Target Recall Comparisons:")
    print(f"   - Target 50% LGG Recall: Closest at tau={row_close_50['threshold_tau_lgg']:.2f} -> LGG Recall={row_close_50['lgg_recall']:.4f}, HGG Recall={row_close_50['hgg_recall']:.4f}")
    print(f"   - Target 60% LGG Recall: Closest at tau={row_close_60['threshold_tau_lgg']:.2f} -> LGG Recall={row_close_60['lgg_recall']:.4f}, HGG Recall={row_close_60['hgg_recall']:.4f}")
    print(f"   - Target 70% LGG Recall: Closest at tau={row_close_70['threshold_tau_lgg']:.2f} -> LGG Recall={row_close_70['lgg_recall']:.4f}, HGG Recall={row_close_70['hgg_recall']:.4f}")
    print(f"   - Target 80% LGG Recall: Closest at tau={row_close_80['threshold_tau_lgg']:.2f} -> LGG Recall={row_close_80['lgg_recall']:.4f}, HGG Recall={row_close_80['hgg_recall']:.4f}")

    print("\n[DONE] Threshold analysis completed successfully.")


if __name__ == "__main__":
    main()
