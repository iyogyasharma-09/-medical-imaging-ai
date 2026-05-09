"""
advanced_analysis.py
--------------------
Runs all 5 advanced analyses in sequence:

  1. Metadata verification     — manifest integrity, label distribution, split leakage
  2. Threshold tuning          — optimal decision threshold per model (ROC/F1 sweep)
  3. Ensemble evaluation       — majority vote + reliability-weighted soft ensemble
  4. Calibration plots         — reliability diagrams + ECE per model
  5. Cross-validation          — 5-fold stratified CV on val+test combined

Results saved to results/advanced_analysis.json
Plots   saved to results/plots/adv_*.png
"""
import json, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix, brier_score_loss,
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_TRAIN_MANIFEST, BRATS2023_VAL_MANIFEST,
    BRATS2023_TEST_MANIFEST, MODELS_DIR, RESULTS_DIR,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_INFO = {
    "resnet18_medicalnet": MODELS_DIR / "resnet18_medicalnet_lgg_hgg_3d.pt",
    "densenet121_3d":      MODELS_DIR / "densenet121_3d_lgg_hgg_3d.pt",
    "swin_transformer_3d": MODELS_DIR / "swin_transformer_3d_lgg_hgg_3d.pt",
}
COLORS = {"resnet18_medicalnet": "#4C72B0",
          "densenet121_3d":      "#DD8452",
          "swin_transformer_3d": "#55A868"}
LABELS = {"resnet18_medicalnet": "ResNet18-MedNet",
          "densenet121_3d":      "DenseNet121-3D",
          "swin_transformer_3d": "SwinTransformer-3D"}

STYLE = {"figure.facecolor":"#0F1117","axes.facecolor":"#1A1D27",
         "axes.edgecolor":"#3A3D4D","axes.labelcolor":"#E0E0E0",
         "xtick.color":"#B0B0B0","ytick.color":"#B0B0B0",
         "text.color":"#E0E0E0","grid.color":"#2A2D3D",
         "grid.linestyle":"--","grid.alpha":0.5,
         "legend.facecolor":"#1A1D27","legend.edgecolor":"#3A3D4D"}
plt.rcParams.update(STYLE)

RESULTS = {}


def savefig(name):
    p = PLOTS_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight",
                facecolor=plt.rcParams["figure.facecolor"])
    plt.close()
    print(f"  Plot: {name}")


def load_model(name):
    ckpt  = torch.load(str(MODELS_INFO[name]), map_location="cpu", weights_only=False)
    model = get_model(name, num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, ckpt


def get_probs(model, df, label_col="label_grade"):
    ds     = BraTS2023Dataset(df, transforms=get_val_transforms(), label_col=label_col)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"]
            with torch.amp.autocast(device_type=DEVICE.type, enabled=True):
                logits = model(imgs)
            y_prob.append(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. METADATA VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
def verify_metadata():
    print("\n" + "="*60)
    print("  1. METADATA VERIFICATION")
    print("="*60)

    manifests = {
        "train": BRATS2023_TRAIN_MANIFEST,
        "val":   BRATS2023_VAL_MANIFEST,
        "test":  BRATS2023_TEST_MANIFEST,
    }
    report = {}
    all_ids = defaultdict(list)

    for split, path in manifests.items():
        df = load_manifest(path, label_col="label_grade")
        df_gli = df[df["task"] == "GLI"].copy()

        n_total   = len(df_gli)
        n_labeled = df_gli["label_grade"].notna().sum()
        n_lgg     = (df_gli["label_grade"] == 0).sum()
        n_hgg     = (df_gli["label_grade"] == 1).sum()
        missing_files = df_gli["data_path"].apply(
            lambda p: not Path(p).exists()
        ).sum()

        # Track subject IDs for leakage check
        for sid in df_gli["subject_id"].tolist():
            all_ids[sid].append(split)

        split_report = {
            "n_subjects":    int(n_total),
            "n_labeled":     int(n_labeled),
            "n_lgg":         int(n_lgg),
            "n_hgg":         int(n_hgg),
            "lgg_frac":      round(n_lgg / n_total, 4) if n_total else 0,
            "missing_files": int(missing_files),
        }
        report[split] = split_report
        print(f"\n  {split.upper()}  (n={n_total})")
        print(f"    Labeled     : {n_labeled}")
        print(f"    LGG / HGG   : {n_lgg} / {n_hgg}  ({n_lgg/n_total*100:.1f}% LGG)")
        print(f"    Missing dirs: {missing_files}")

    # Leakage check
    leaked = {sid: splits for sid, splits in all_ids.items() if len(splits) > 1}
    report["leakage"] = {"n_leaked": len(leaked), "leaked_ids": list(leaked.keys())[:10]}
    print(f"\n  Leakage check: {len(leaked)} subjects appear in multiple splits "
          f"{'[WARNING]' if leaked else '[OK]'}")

    RESULTS["metadata"] = report
    print("  [DONE] Metadata verification complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. THRESHOLD TUNING
# ═══════════════════════════════════════════════════════════════════════════════
def threshold_tuning():
    print("\n" + "="*60)
    print("  2. THRESHOLD TUNING")
    print("="*60)

    df_val  = load_manifest(BRATS2023_VAL_MANIFEST,  label_col="label_grade")
    df_val  = df_val[df_val["task"] == "GLI"].reset_index(drop=True)
    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test[df_test["task"] == "GLI"].reset_index(drop=True)

    report   = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Threshold Tuning — F1 & Sensitivity/Specificity vs Threshold",
                 fontsize=13, fontweight="bold")

    for ax, mname in zip(axes, MODELS_INFO):
        model, _ = load_model(mname)
        y_val,  p_val  = get_probs(model, df_val)
        y_test, p_test = get_probs(model, df_test)
        del model; torch.cuda.empty_cache()

        # Sweep thresholds on val, pick best F1
        thresholds = np.linspace(0.1, 0.9, 81)
        f1s, sens, spec = [], [], []
        for thr in thresholds:
            pred = (p_val[:, 1] >= thr).astype(int)
            f1s.append(f1_score(y_val, pred, average="weighted", zero_division=0))
            cm   = confusion_matrix(y_val, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens.append(tp / (tp + fn) if (tp + fn) else 0)
            spec.append(tn / (tn + fp) if (tn + fp) else 0)

        best_idx = int(np.argmax(f1s))
        best_thr = float(thresholds[best_idx])

        # Apply best threshold to test set
        pred_test = (p_test[:, 1] >= best_thr).astype(int)
        test_acc  = float(accuracy_score(y_test, pred_test))
        test_f1   = float(f1_score(y_test, pred_test, average="weighted", zero_division=0))
        test_auc  = float(roc_auc_score(y_test, p_test[:, 1]))
        cm_test   = confusion_matrix(y_test, pred_test, labels=[0, 1])
        tn, fp, fn, tp = cm_test.ravel()
        test_sens = float(tp / (tp + fn)) if (tp + fn) else 0
        test_spec = float(tn / (tn + fp)) if (tn + fp) else 0

        report[mname] = {
            "best_threshold":       best_thr,
            "val_best_f1":          float(f1s[best_idx]),
            "test_acc_at_best_thr": test_acc,
            "test_f1_at_best_thr":  test_f1,
            "test_auc":             test_auc,
            "test_sensitivity":     test_sens,
            "test_specificity":     test_spec,
        }
        print(f"\n  {LABELS[mname]}")
        print(f"    Best threshold: {best_thr:.2f}  (val F1={f1s[best_idx]:.4f})")
        print(f"    Test @ thr     : acc={test_acc:.4f}  sens={test_sens:.4f}  spec={test_spec:.4f}")

        # Plot
        c = COLORS[mname]
        ax.plot(thresholds, f1s,  color=c,       lw=2.5, label="F1 (weighted)")
        ax.plot(thresholds, sens, color=c, lw=2, ls="--", alpha=0.7, label="Sensitivity")
        ax.plot(thresholds, spec, color=c, lw=2, ls=":",  alpha=0.7, label="Specificity")
        ax.axvline(best_thr, color="white", lw=1, ls="--", alpha=0.6,
                   label=f"Best thr={best_thr:.2f}")
        ax.set_title(LABELS[mname], color=c, fontweight="bold")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    savefig("adv_02_threshold_tuning.png")
    RESULTS["threshold_tuning"] = report
    print("  [DONE] Threshold tuning complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ENSEMBLE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def ensemble_evaluation():
    print("\n" + "="*60)
    print("  3. ENSEMBLE EVALUATION")
    print("="*60)

    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test[df_test["task"] == "GLI"].reset_index(drop=True)

    all_probs = {}
    for mname in MODELS_INFO:
        model, ckpt = load_model(mname)
        y_true, probs = get_probs(model, df_test)
        all_probs[mname] = probs
        del model; torch.cuda.empty_cache()

    # Weights = val accuracy from registry
    val_accs = {
        "resnet18_medicalnet": 0.9465,
        "densenet121_3d":      0.9465,
        "swin_transformer_3d": 0.9465,
    }
    # AUC-weighted (better discrimination)
    auc_weights = {
        "resnet18_medicalnet": 0.779,
        "densenet121_3d":      0.883,
        "swin_transformer_3d": 0.885,
    }
    total_w = sum(auc_weights.values())

    # Soft ensemble (AUC-weighted)
    soft_prob = sum(auc_weights[m] / total_w * all_probs[m] for m in MODELS_INFO)
    soft_pred = soft_prob.argmax(axis=1)

    # Majority vote
    preds_all  = np.stack([all_probs[m].argmax(axis=1) for m in MODELS_INFO])
    vote_pred  = (preds_all.sum(axis=0) >= 2).astype(int)   # 2/3 majority

    report = {}
    for name, pred, prob in [
        ("soft_ensemble", soft_pred, soft_prob),
        ("majority_vote", vote_pred, soft_prob),  # use soft prob for AUC
    ]:
        acc  = float(accuracy_score(y_true, pred))
        f1   = float(f1_score(y_true, pred, average="weighted", zero_division=0))
        auc  = float(roc_auc_score(y_true, prob[:, 1]))
        cm   = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) else 0
        spec = float(tn / (tn + fp)) if (tn + fp) else 0
        report[name] = {"accuracy": acc, "f1_weighted": f1, "auc": auc,
                        "sensitivity": sens, "specificity": spec,
                        "confusion_matrix": cm.tolist()}
        print(f"\n  {name}")
        print(f"    Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  Sens={sens:.4f}  Spec={spec:.4f}")

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("ROC Curves — Individual Models + Ensemble", fontweight="bold")
    for mname in MODELS_INFO:
        fpr, tpr, _ = roc_curve(y_true, all_probs[mname][:, 1])
        auc = roc_auc_score(y_true, all_probs[mname][:, 1])
        ax.plot(fpr, tpr, color=COLORS[mname], lw=2,
                label=f"{LABELS[mname]} (AUC={auc:.3f})")
    fpr_e, tpr_e, _ = roc_curve(y_true, soft_prob[:, 1])
    auc_e = roc_auc_score(y_true, soft_prob[:, 1])
    ax.plot(fpr_e, tpr_e, color="white", lw=2.5, ls="--",
            label=f"Soft Ensemble (AUC={auc_e:.3f})")
    ax.plot([0,1],[0,1], "gray", lw=1, ls=":")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=9); ax.grid(True)
    plt.tight_layout()
    savefig("adv_03_roc_curves.png")

    RESULTS["ensemble"] = report
    print("  [DONE] Ensemble evaluation complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
def calibration_plots():
    print("\n" + "="*60)
    print("  4. CALIBRATION PLOTS")
    print("="*60)

    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test[df_test["task"] == "GLI"].reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Calibration (Reliability Diagrams) — Test Set", fontsize=13, fontweight="bold")
    report = {}

    for ax, mname in zip(axes, MODELS_INFO):
        model, _ = load_model(mname)
        y_true, probs = get_probs(model, df_test)
        del model; torch.cuda.empty_cache()

        prob_pos = probs[:, 1]
        frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=10, strategy="uniform")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        brier = float(brier_score_loss(y_true, prob_pos))

        report[mname] = {"ece": ece, "brier_score": brier}
        print(f"  {LABELS[mname]:28s}  ECE={ece:.4f}  Brier={brier:.4f}")

        c = COLORS[mname]
        ax.plot([0,1],[0,1], "gray", lw=1, ls="--", label="Perfect")
        ax.plot(mean_pred, frac_pos, color=c, lw=2.5, marker="o", ms=6,
                label=f"ECE={ece:.4f}  Brier={brier:.4f}")
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        color=c, alpha=0.15, label="Calibration gap")
        ax.set_title(LABELS[mname], color=c, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives (HGG)")
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    savefig("adv_04_calibration.png")
    RESULTS["calibration"] = report
    print("  [DONE] Calibration plots complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION (5-fold stratified on val+test combined)
# ═══════════════════════════════════════════════════════════════════════════════
def cross_validation():
    print("\n" + "="*60)
    print("  5. CROSS-VALIDATION (5-fold, val+test combined, inference only)")
    print("="*60)

    # Use val+test combined (375 subjects) for CV — models are fixed (no retraining)
    import pandas as pd
    df_val  = load_manifest(BRATS2023_VAL_MANIFEST,  label_col="label_grade")
    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_all  = pd.concat([
        df_val[df_val["task"] == "GLI"],
        df_test[df_test["task"] == "GLI"],
    ]).reset_index(drop=True)

    labels_all = df_all["label_grade"].values.astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    report = {}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("5-Fold CV (Inference) — Val+Test Combined", fontsize=13, fontweight="bold")

    for ax, mname in zip(axes, MODELS_INFO):
        model, _ = load_model(mname)
        # Get probs for entire val+test pool
        y_all, p_all = get_probs(model, df_all)
        del model; torch.cuda.empty_cache()

        fold_accs, fold_aucs, fold_f1s = [], [], []
        for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(y_all)), y_all)):
            y_f = y_all[test_idx]
            p_f = p_all[test_idx]
            fold_accs.append(accuracy_score(y_f, p_f.argmax(1)))
            fold_f1s.append(f1_score(y_f, p_f.argmax(1), average="weighted", zero_division=0))
            try:    fold_aucs.append(roc_auc_score(y_f, p_f[:, 1]))
            except: fold_aucs.append(0.0)

        mean_acc = float(np.mean(fold_accs))
        std_acc  = float(np.std(fold_accs))
        mean_auc = float(np.mean(fold_aucs))
        std_auc  = float(np.std(fold_aucs))
        mean_f1  = float(np.mean(fold_f1s))
        std_f1   = float(np.std(fold_f1s))

        report[mname] = {
            "acc_mean": mean_acc, "acc_std": std_acc,
            "auc_mean": mean_auc, "auc_std": std_auc,
            "f1_mean":  mean_f1,  "f1_std":  std_f1,
            "fold_accs": [float(x) for x in fold_accs],
            "fold_aucs": [float(x) for x in fold_aucs],
        }
        print(f"\n  {LABELS[mname]}")
        print(f"    Acc : {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"    AUC : {mean_auc:.4f} +/- {std_auc:.4f}")
        print(f"    F1  : {mean_f1:.4f}  +/- {std_f1:.4f}")

        c = COLORS[mname]
        folds = list(range(1, 6))
        ax.plot(folds, fold_accs, color=c, lw=2, marker="o", ms=7, label="Accuracy")
        ax.plot(folds, fold_aucs, color=c, lw=2, marker="s", ms=7, ls="--", alpha=0.7, label="AUC")
        ax.axhline(mean_acc, color=c, lw=1, ls=":", alpha=0.5,
                   label=f"Mean acc={mean_acc:.3f}")
        ax.set_title(LABELS[mname], color=c, fontweight="bold")
        ax.set_xlabel("Fold"); ax.set_ylabel("Score")
        ax.set_xticks(folds); ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    savefig("adv_05_crossval.png")
    RESULTS["cross_validation"] = report
    print("  [DONE] Cross-validation complete.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nDevice: {DEVICE}")
    print("Running 5 advanced analyses...\n")

    verify_metadata()
    threshold_tuning()
    ensemble_evaluation()
    calibration_plots()
    cross_validation()

    out = RESULTS_DIR / "advanced_analysis.json"
    out.write_text(json.dumps(RESULTS, indent=2))
    print(f"\n\nAll results saved: {out}")
    print(f"All plots saved:   {PLOTS_DIR}")
