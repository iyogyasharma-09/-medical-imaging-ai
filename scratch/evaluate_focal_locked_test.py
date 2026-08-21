"""
evaluate_focal_locked_test.py
------------------------------
Standalone evaluation of the Focal Loss ResNet18 MedicalNet checkpoint
on the SAME locked BraTS 2023 GLI test cohort used for the CE baseline.

Safety:
- Inference only (model.eval(), torch.no_grad())
- No parameter updates, no optimizer step
- No threshold tuning on test set
- Outputs isolated to results/focal_locked_test/ (no baseline overwrite)
- Uses identical infrastructure: get_model, get_val_transforms, load_manifest

Methodology match with CE baseline:
- Same test manifest (188 GLI subjects: 12 LGG, 176 HGG)
- Same preprocessing (get_val_transforms)
- Same threshold (0.5 via argmax)
- Same model architecture (get_model from train_brats2023.py)
- Same AMP inference
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import BRATS2023_TEST_MANIFEST, RESULTS_DIR
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error — identical to evaluate_models.py."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop_in_bin
    return float(ece)


def main():
    start_time = time.time()

    print("=" * 70)
    print("FOCAL LOSS — LOCKED TEST EVALUATION")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load SAME locked test cohort ---
    df_test = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test[df_test["task"] == "GLI"].reset_index(drop=True)
    ds_test = BraTS2023Dataset(
        df_test, transforms=get_val_transforms(),
        label_col="label_grade", return_meta=True,
    )
    test_loader = DataLoader(
        ds_test, batch_size=4, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    n_lgg = int(sum(df_test["label_grade"] == 0))
    n_hgg = int(sum(df_test["label_grade"] == 1))
    print(f"\nTest subjects: {len(df_test)} (LGG={n_lgg}, HGG={n_hgg})")

    # --- Load Focal checkpoint ---
    ckpt_path = BASE_DIR / "models" / "resnet18_medicalnet_focal_lgg_hgg_3d.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Focal checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"\nCheckpoint: {ckpt_path.name}")
    print(f"  model_name:        {ckpt.get('model_name')}")
    print(f"  task:              {ckpt.get('task')}")
    print(f"  label_col:         {ckpt.get('label_col')}")
    print(f"  training_protocol: {ckpt.get('training_protocol')}")
    print(f"  focal_alpha:       {ckpt.get('focal_alpha')}")
    print(f"  focal_gamma:       {ckpt.get('focal_gamma')}")
    print(f"  num_classes:       {ckpt.get('num_classes')}")
    print(f"  input_shape:       {ckpt.get('input_shape')}")
    print(f"  spatial_size:      {ckpt.get('spatial_size')}")
    print(f"  best_val_macro_f1: {ckpt.get('best_val_macro_f1')}")
    print(f"  best_val_lgg_rec:  {ckpt.get('best_val_lgg_recall')}")

    # --- Build model and load weights ---
    model = get_model("resnet18_medicalnet", num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # --- Inference (identical to evaluate_locked_test.py) ---
    print("\nRunning inference...")
    subject_ids, y_true_list, y_prob_0_list, y_prob_1_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Focal Inference"):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            s_ids = batch.get("subject_id", [f"sub_{i}" for i in range(len(imgs))])

            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)

            subject_ids.extend(s_ids)
            y_true_list.extend(labels.cpu().numpy())
            probs_np = probs.cpu().numpy()
            y_prob_0_list.extend(probs_np[:, 0])
            y_prob_1_list.extend(probs_np[:, 1])

    y_true = np.array(y_true_list)
    y_prob_0 = np.array(y_prob_0_list)
    y_prob_1 = np.array(y_prob_1_list)
    y_pred = (y_prob_1 >= 0.5).astype(int)  # Same threshold as CE baseline

    # --- Metrics (identical to baseline methodology) ---
    acc = float(accuracy_score(y_true, y_pred))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob_1))
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    lgg_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    hgg_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # ECE (from evaluate_models.py compute_ece)
    y_prob_matrix = np.column_stack([y_prob_0, y_prob_1])
    ece = compute_ece(y_true, y_prob_matrix)

    # Brier score
    brier = float(np.mean((y_prob_1 - y_true) ** 2))

    # Predictive entropy (uncertainty)
    p0c = np.clip(y_prob_0, 1e-12, 1.0)
    p1c = np.clip(y_prob_1, 1e-12, 1.0)
    uncertainty = -(p0c * np.log2(p0c) + p1c * np.log2(p1c))

    # --- Print results ---
    print(f"\n{'='*70}")
    print(f"FOCAL LOSS ResNet18 MedicalNet — LOCKED TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Subjects:     {len(y_true)} (LGG={tn+fp}, HGG={fn+tp})")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Macro-F1:     {f1_m:.4f}")
    print(f"  Weighted-F1:  {f1_w:.4f}")
    print(f"  AUC:          {auc:.4f}")
    print(f"  LGG Recall:   {lgg_recall:.4f}  ({tn}/{tn+fp})")
    print(f"  HGG Recall:   {hgg_recall:.4f}  ({tp}/{tp+fn})")
    print(f"  Precision:    {precision_val:.4f}")
    print(f"  ECE:          {ece:.4f}")
    print(f"  Brier Score:  {brier:.4f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # --- Save per-subject predictions ---
    confidence_arr = np.maximum(y_prob_0, y_prob_1)
    df_preds = pd.DataFrame({
        "subject_id": subject_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "prob_lgg": y_prob_0,
        "prob_hgg": y_prob_1,
        "confidence": confidence_arr,
        "uncertainty": uncertainty,
    })

    out_dir = RESULTS_DIR / "focal_locked_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "resnet18_focal_test_predictions.csv"
    df_preds.to_csv(pred_path, index=False)
    print(f"\n  Predictions saved: {pred_path}")

    # --- Save metrics JSON ---
    metrics = {
        "model_key": "resnet18_medicalnet",
        "model_name": "ResNet18 MedicalNet",
        "loss_function": "Focal Loss",
        "focal_alpha": float(ckpt.get("focal_alpha", 1.0)),
        "focal_gamma": float(ckpt.get("focal_gamma", 2.0)),
        "training_protocol": "FOCAL",
        "checkpoint_path": str(ckpt_path),
        "n_test_subjects": len(y_true),
        "lgg_subjects": int(tn + fp),
        "hgg_subjects": int(fn + tp),
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "lgg_recall": round(lgg_recall, 4),
        "hgg_recall": round(hgg_recall, 4),
        "precision": round(precision_val, 4),
        "auc": round(auc, 4),
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "matrix": cm.tolist(),
        },
    }
    metrics_path = out_dir / "resnet18_focal_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics saved: {metrics_path}")

    # --- Save comparison JSON (if baseline exists) ---
    baseline_metrics_path = RESULTS_DIR / "locked_test" / "protocol_A" / "resnet18_metrics.json"
    if baseline_metrics_path.exists():
        baseline = json.loads(baseline_metrics_path.read_text())
        comparison = {
            "baseline_CE_protocol_A": baseline,
            "focal_loss": metrics,
            "deltas": {
                "accuracy": round(metrics["accuracy"] - baseline["accuracy"], 4),
                "f1_macro": round(metrics["f1_macro"] - baseline["f1_macro"], 4),
                "lgg_recall": round(metrics["lgg_recall"] - baseline["lgg_recall"], 4),
                "hgg_recall": round(metrics["hgg_recall"] - baseline.get("hgg_recall", 0), 4),
                "auc": round(metrics["auc"] - baseline["auc"], 4),
            },
        }
        comp_path = out_dir / "focal_vs_baseline_comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2))
        print(f"  Comparison saved: {comp_path}")

        print(f"\n{'='*70}")
        print(f"CE BASELINE (Protocol A) vs FOCAL LOSS — DELTA SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Metric':<16s} {'CE Baseline':>12s} {'Focal':>12s} {'Delta':>12s}")
        print(f"  {'-'*52}")
        for k in ["accuracy", "f1_macro", "lgg_recall", "hgg_recall", "auc"]:
            ce_val = baseline.get(k, 0)
            fo_val = metrics.get(k, 0)
            delta = comparison["deltas"][k]
            sign = "+" if delta > 0 else ""
            print(f"  {k:<16s} {ce_val:>12.4f} {fo_val:>12.4f} {sign}{delta:>11.4f}")
    else:
        print(f"\n  [WARN] Baseline metrics not found at {baseline_metrics_path}")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Evaluation completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"All Focal evaluation artifacts saved to: {out_dir}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
