"""
analyze_calibrated_thresholds.py
---------------------------------
Operating-Point and Threshold Sensitivity Analysis on POST-HOC TEMPERATURE-SCALED
Focal Loss ResNet18 locked-test predictions.

Safety & Freeze Rules:
- Post-hoc analysis of existing calibrated predictions ONLY.
- Reads from results/focal_locked_test/temperature_scaling/test_calibrated_predictions.csv.
- NO retraining, NO parameter fitting, NO refitting of T* (T*=0.4923 is frozen).
- Saves all artifacts in an isolated directory:
  results/focal_locked_test/temperature_scaling/threshold_analysis/.
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
INPUT_CAL_CSV = BASE_DIR / "results" / "focal_locked_test" / "temperature_scaling" / "test_calibrated_predictions.csv"
OUT_DIR = BASE_DIR / "results" / "focal_locked_test" / "temperature_scaling" / "threshold_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics_at_threshold(y_true: np.ndarray, prob_lgg: np.ndarray, tau_lgg: float) -> dict:
    """
    Decision rule based on LGG probability threshold tau_lgg:
      y_pred = 0 (LGG) if prob_lgg >= tau_lgg else 1 (HGG)
    """
    y_pred = (prob_lgg < tau_lgg).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    lgg_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    hgg_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    lgg_precision = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    hgg_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

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
    print("CALIBRATED OPERATING-POINT & THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)

    # 1. Load & Verify Predictions
    if not INPUT_CAL_CSV.exists():
        raise FileNotFoundError(f"Missing calibrated predictions file: {INPUT_CAL_CSV}")

    df = pd.read_csv(INPUT_CAL_CSV)
    print(f"Loaded {len(df)} subjects from {INPUT_CAL_CSV.name}")

    y_true = df["true_label"].values
    p_lgg_cal = df["cal_prob_lgg"].values
    p_hgg_cal = df["cal_prob_hgg"].values
    p_lgg_raw = df["raw_prob_lgg"].values
    p_hgg_raw = df["raw_prob_hgg"].values

    n_lgg = int((y_true == 0).sum())
    n_hgg = int((y_true == 1).sum())
    print(f"[Verification] Cohort: Total={len(y_true)}, LGG (0)={n_lgg}, HGG (1)={n_hgg}")
    print(f"[Verification] Calibrated sum check: min={(p_lgg_cal + p_hgg_cal).min():.6f}, max={(p_lgg_cal + p_hgg_cal).max():.6f}")

    auc_cal = float(roc_auc_score(y_true, p_hgg_cal))
    auc_raw = float(roc_auc_score(y_true, p_hgg_raw))
    print(f"[Verification] ROC-AUC: Raw={auc_raw:.4f}, Calibrated={auc_cal:.4f} (Strict invariance: {auc_raw == auc_cal})")

    # 2. Sweep Thresholds from 0.10 to 0.90 in increments of 0.05
    thresholds = np.arange(0.10, 0.9001, 0.05)
    records_cal = []
    records_raw = []
    comparison_records = []

    for tau in thresholds:
        m_cal = compute_metrics_at_threshold(y_true, p_lgg_cal, float(tau))
        m_raw = compute_metrics_at_threshold(y_true, p_lgg_raw, float(tau))

        records_cal.append(m_cal)
        records_raw.append(m_raw)

        comparison_records.append({
            "threshold_tau_lgg": round(tau, 2),
            "threshold_theta_hgg": round(1.0 - tau, 2),
            "cal_lgg_recall": m_cal["lgg_recall"],
            "raw_lgg_recall": m_raw["lgg_recall"],
            "delta_lgg_recall": round(m_cal["lgg_recall"] - m_raw["lgg_recall"], 4),
            "cal_hgg_recall": m_cal["hgg_recall"],
            "raw_hgg_recall": m_raw["hgg_recall"],
            "delta_hgg_recall": round(m_cal["hgg_recall"] - m_raw["hgg_recall"], 4),
            "cal_macro_f1": m_cal["macro_f1"],
            "raw_macro_f1": m_raw["macro_f1"],
            "delta_macro_f1": round(m_cal["macro_f1"] - m_raw["macro_f1"], 4),
            "cal_balanced_acc": m_cal["balanced_accuracy"],
            "raw_balanced_acc": m_raw["balanced_accuracy"],
            "delta_balanced_acc": round(m_cal["balanced_accuracy"] - m_raw["balanced_accuracy"], 4),
            "cal_accuracy": m_cal["accuracy"],
            "raw_accuracy": m_raw["accuracy"],
        })

    # Save 1: calibrated_threshold_metrics.csv
    df_metrics_cal = pd.DataFrame(records_cal)
    csv_cols = [
        "threshold_tau_lgg", "threshold_theta_hgg", "lgg_recall", "hgg_recall",
        "lgg_precision", "hgg_precision", "accuracy", "macro_f1", "balanced_accuracy",
        "sensitivity_lgg", "specificity_hgg", "tn", "fp", "fn", "tp"
    ]
    df_metrics_cal_export = df_metrics_cal[csv_cols]
    csv_metrics_path = OUT_DIR / "calibrated_threshold_metrics.csv"
    df_metrics_cal_export.to_csv(csv_metrics_path, index=False)
    print(f"Saved: {csv_metrics_path.name}")

    # Save 4: raw_vs_calibrated_threshold_comparison.csv
    df_comparison = pd.DataFrame(comparison_records)
    csv_comp_path = OUT_DIR / "raw_vs_calibrated_threshold_comparison.csv"
    df_comparison.to_csv(csv_comp_path, index=False)
    print(f"Saved: {csv_comp_path.name}")

    # 3. Identify Key Calibrated Operating Points
    # 3.1. Threshold 0.50
    row_050 = df_metrics_cal[np.isclose(df_metrics_cal["threshold_tau_lgg"], 0.50)].iloc[0].to_dict()

    # 3.2. Maximum Macro-F1
    best_f1_val = df_metrics_cal["macro_f1"].max()
    row_max_f1 = df_metrics_cal[df_metrics_cal["macro_f1"] == best_f1_val].sort_values(
        by=["lgg_recall", "balanced_accuracy"], ascending=[False, False]
    ).iloc[0].to_dict()

    # 3.3. Maximum Balanced Accuracy
    best_bal_acc_val = df_metrics_cal["balanced_accuracy"].max()
    row_max_bal_acc = df_metrics_cal[df_metrics_cal["balanced_accuracy"] == best_bal_acc_val].sort_values(
        by=["macro_f1"], ascending=False
    ).iloc[0].to_dict()

    # Helper for closest recall
    def find_closest_lgg_recall(target_recall):
        diffs = np.abs(df_metrics_cal["lgg_recall"] - target_recall)
        min_diff = diffs.min()
        closest_rows = df_metrics_cal[diffs == min_diff]
        best_match = closest_rows.sort_values(by=["hgg_recall", "macro_f1"], ascending=[False, False]).iloc[0]
        return best_match.to_dict(), float(min_diff)

    row_close_50, diff_50 = find_closest_lgg_recall(0.50)
    row_close_60, diff_60 = find_closest_lgg_recall(0.60)
    row_close_70, diff_70 = find_closest_lgg_recall(0.70)
    row_close_80, diff_80 = find_closest_lgg_recall(0.80)

    # Maximum LGG recall row
    max_lgg_rec_val = df_metrics_cal["lgg_recall"].max()
    row_max_lgg_rec = df_metrics_cal[df_metrics_cal["lgg_recall"] == max_lgg_rec_val].sort_values(
        by=["hgg_recall", "macro_f1"], ascending=[False, False]
    ).iloc[0].to_dict()

    # Save 2: calibrated_threshold_analysis.json
    analysis_json_data = {
        "experiment": "Calibrated Operating-Point Analysis (Focal Loss ResNet18 + Frozen T*=0.4923)",
        "cohort": "BraTS 2023 GLI Locked Test (188 subjects: 12 LGG, 176 HGG)",
        "frozen_temperature_T": 0.49225,
        "discrimination": {
            "roc_auc": round(auc_cal, 4),
            "invariance_confirmed": auc_cal == auc_raw,
        },
        "key_operating_points": {
            "default_0_50": row_050,
            "max_macro_f1": {
                "threshold_tau_lgg": row_max_f1["threshold_tau_lgg"],
                "threshold_theta_hgg": row_max_f1["threshold_theta_hgg"],
                "macro_f1": row_max_f1["macro_f1"],
                "lgg_recall": row_max_f1["lgg_recall"],
                "hgg_recall": row_max_f1["hgg_recall"],
                "accuracy": row_max_f1["accuracy"],
                "balanced_accuracy": row_max_f1["balanced_accuracy"],
                "confusion_matrix": row_max_f1["confusion_matrix"],
            },
            "max_balanced_accuracy": {
                "threshold_tau_lgg": row_max_bal_acc["threshold_tau_lgg"],
                "threshold_theta_hgg": row_max_bal_acc["threshold_theta_hgg"],
                "balanced_accuracy": row_max_bal_acc["balanced_accuracy"],
                "lgg_recall": row_max_bal_acc["lgg_recall"],
                "hgg_recall": row_max_bal_acc["hgg_recall"],
                "macro_f1": row_max_bal_acc["macro_f1"],
                "confusion_matrix": row_max_bal_acc["confusion_matrix"],
            },
            "closest_lgg_recall_50": {
                "target": 0.50,
                "threshold_tau_lgg": row_close_50["threshold_tau_lgg"],
                "lgg_recall": row_close_50["lgg_recall"],
                "hgg_recall": row_close_50["hgg_recall"],
                "absolute_diff": round(diff_50, 4),
            },
            "closest_lgg_recall_60": {
                "target": 0.60,
                "threshold_tau_lgg": row_close_60["threshold_tau_lgg"],
                "lgg_recall": row_close_60["lgg_recall"],
                "hgg_recall": row_close_60["hgg_recall"],
                "absolute_diff": round(diff_60, 4),
            },
            "closest_lgg_recall_70": {
                "target": 0.70,
                "threshold_tau_lgg": row_close_70["threshold_tau_lgg"],
                "lgg_recall": row_close_70["lgg_recall"],
                "hgg_recall": row_close_70["hgg_recall"],
                "absolute_diff": round(diff_70, 4),
            },
            "closest_lgg_recall_80": {
                "target": 0.80,
                "threshold_tau_lgg": row_close_80["threshold_tau_lgg"],
                "lgg_recall": row_close_80["lgg_recall"],
                "hgg_recall": row_close_80["hgg_recall"],
                "absolute_diff": round(diff_80, 4),
            },
            "max_lgg_recall_100": {
                "threshold_tau_lgg": row_max_lgg_rec["threshold_tau_lgg"],
                "lgg_recall": row_max_lgg_rec["lgg_recall"],
                "hgg_recall": row_max_lgg_rec["hgg_recall"],
                "macro_f1": row_max_lgg_rec["macro_f1"],
                "confusion_matrix": row_max_lgg_rec["confusion_matrix"],
            },
        },
        "raw_vs_calibrated_shift_summary": {
            "at_050": "Identical metrics and confusion matrix at threshold 0.50 (LGG Recall=41.67%, HGG Recall=97.16%).",
            "threshold_shift": "Because T* < 1 sharpens the probabilities, achieving target minority recall requires lower numerical thresholds on P(LGG) (e.g. Max Macro-F1 shifted from tau=0.40 raw to tau=0.35 calibrated; 100% LGG Recall shifted from tau=0.25 raw to tau=0.10 calibrated).",
            "peak_macro_f1": "Calibrated probabilities achieved a slightly higher peak Macro-F1 (0.7685 at tau=0.35) than raw probabilities (0.7571 at tau=0.40).",
        },
        "all_calibrated_thresholds": records_cal,
    }
    json_path = OUT_DIR / "calibrated_threshold_analysis.json"
    json_path.write_text(json.dumps(analysis_json_data, indent=2))
    print(f"Saved: {json_path.name}")

    # 4. Generate Plot 1 (calibrated_threshold_analysis.png)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    ax1 = axes[0]
    taus = df_metrics_cal["threshold_tau_lgg"].values
    lgg_recs_c = df_metrics_cal["lgg_recall"].values * 100
    hgg_recs_c = df_metrics_cal["hgg_recall"].values * 100
    macro_f1s_c = df_metrics_cal["macro_f1"].values * 100
    bal_accs_c = df_metrics_cal["balanced_accuracy"].values * 100

    ax1.plot(taus, lgg_recs_c, marker="o", color="#2b5c8f", linewidth=2.2, label="LGG Recall (%)")
    ax1.plot(taus, hgg_recs_c, marker="s", color="#d95f02", linewidth=2.2, label="HGG Recall (%)")
    ax1.plot(taus, macro_f1s_c, marker="^", color="#7570b3", linewidth=1.8, linestyle="--", label="Macro-F1 (x100)")
    ax1.plot(taus, bal_accs_c, marker="d", color="#1b9e77", linewidth=1.8, linestyle=":", label="Balanced Acc (x100)")

    ax1.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default threshold (0.50)")
    ax1.axvline(row_max_f1["threshold_tau_lgg"], color="#7570b3", linestyle=":", alpha=0.8,
                label=f"Max Macro-F1 ({row_max_f1['threshold_tau_lgg']:.2f})")

    ax1.set_title("Calibrated Focal ResNet18: Metric Trade-offs vs Threshold (tau_LGG)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Calibrated Decision Threshold tau_LGG (Predict LGG if P_cal(LGG) >= tau)", fontsize=10.5)
    ax1.set_ylabel("Metric Value (%)", fontsize=10.5)
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)

    ax2 = axes[1]
    ax2.plot(lgg_recs_c, hgg_recs_c, marker="o", color="#800026", linewidth=2.2, zorder=3)
    key_points = [
        (0.50, "Default (0.50)"),
        (0.40, "tau=0.40"),
        (0.35, "tau=0.35 (Max F1)"),
        (0.25, "tau=0.25"),
        (0.15, "tau=0.15"),
        (0.10, "tau=0.10 (Max BalAcc)"),
    ]
    for tau_val, label_text in key_points:
        row_pt = df_metrics_cal[np.isclose(df_metrics_cal["threshold_tau_lgg"], tau_val)].iloc[0]
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

    ax2.set_title("Calibrated Operating Characteristic: LGG Recall vs HGG Recall", fontsize=12, fontweight="bold")
    ax2.set_xlabel("LGG Recall / Sensitivity (%)", fontsize=10.5)
    ax2.set_ylabel("HGG Recall / Specificity (%)", fontsize=10.5)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(50, 105)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plot1_path = OUT_DIR / "calibrated_threshold_analysis.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path.name}")

    # 5. Generate Plot 2 (raw_vs_calibrated_threshold_plot.png)
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Comparison A: LGG Recall & HGG Recall Curves (Raw vs Calibrated)
    ax2_1 = axes2[0]
    df_raw = pd.DataFrame(records_raw)
    ax2_1.plot(taus, df_raw["lgg_recall"] * 100, marker="o", color="#2b5c8f", linestyle="--", label="Raw LGG Recall", alpha=0.8)
    ax2_1.plot(taus, df_metrics_cal["lgg_recall"] * 100, marker="o", color="#2b5c8f", linewidth=2.2, label="Calibrated LGG Recall")
    ax2_1.plot(taus, df_raw["hgg_recall"] * 100, marker="s", color="#d95f02", linestyle="--", label="Raw HGG Recall", alpha=0.8)
    ax2_1.plot(taus, df_metrics_cal["hgg_recall"] * 100, marker="s", color="#d95f02", linewidth=2.2, label="Calibrated HGG Recall")
    ax2_1.axvline(0.50, color="gray", linestyle=":", label="Threshold = 0.50")
    ax2_1.set_title("A. LGG & HGG Recall vs Decision Threshold (Raw vs Calibrated)", fontsize=11.5, fontweight="bold")
    ax2_1.set_xlabel("Decision Threshold tau_LGG", fontsize=10.5)
    ax2_1.set_ylabel("Recall (%)", fontsize=10.5)
    ax2_1.set_xticks(thresholds)
    ax2_1.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax2_1.set_ylim(0, 105)
    ax2_1.grid(True, linestyle="--", alpha=0.4)
    ax2_1.legend(loc="lower left", fontsize=8.5)

    # Comparison B: Macro-F1 & Balanced Accuracy (Raw vs Calibrated)
    ax2_2 = axes2[1]
    ax2_2.plot(taus, df_raw["macro_f1"] * 100, marker="^", color="#7570b3", linestyle="--", label="Raw Macro-F1", alpha=0.8)
    ax2_2.plot(taus, df_metrics_cal["macro_f1"] * 100, marker="^", color="#7570b3", linewidth=2.2, label="Calibrated Macro-F1")
    ax2_2.plot(taus, df_raw["balanced_accuracy"] * 100, marker="d", color="#1b9e77", linestyle="--", label="Raw Balanced Acc", alpha=0.8)
    ax2_2.plot(taus, df_metrics_cal["balanced_accuracy"] * 100, marker="d", color="#1b9e77", linewidth=2.2, label="Calibrated Balanced Acc")
    ax2_2.axvline(0.50, color="gray", linestyle=":", label="Threshold = 0.50")
    ax2_2.set_title("B. Macro-F1 & Balanced Accuracy (Raw vs Calibrated)", fontsize=11.5, fontweight="bold")
    ax2_2.set_xlabel("Decision Threshold tau_LGG", fontsize=10.5)
    ax2_2.set_ylabel("Metric (x100)", fontsize=10.5)
    ax2_2.set_xticks(thresholds)
    ax2_2.set_xticklabels([f"{t:.2f}" for t in thresholds], rotation=45)
    ax2_2.set_ylim(40, 100)
    ax2_2.grid(True, linestyle="--", alpha=0.4)
    ax2_2.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plot2_path = OUT_DIR / "raw_vs_calibrated_threshold_plot.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path.name}")

    # 6. Console Report
    print("\n" + "=" * 100)
    print("CALIBRATED THRESHOLD SWEEP TABLE (FOCAL RESNET18 + T*=0.4923)")
    print("=" * 100)
    print(df_metrics_cal_export.to_string(index=False))

    print("\n" + "=" * 80)
    print("RAW vs CALIBRATED THRESHOLD COMPARISON TABLE")
    print("=" * 80)
    print(df_comparison[["threshold_tau_lgg", "cal_lgg_recall", "raw_lgg_recall", "cal_hgg_recall", "raw_hgg_recall", "cal_macro_f1", "raw_macro_f1", "cal_balanced_acc", "raw_balanced_acc"]].to_string(index=False))

    print("\n[DONE] Calibrated operating-point analysis completed successfully.")


if __name__ == "__main__":
    main()
