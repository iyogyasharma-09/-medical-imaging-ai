"""
evaluate_models.py
------------------
Full PyTorch GPU evaluation of all trained models on:
  - Validation set  (187 GLI subjects)
  - Test set        (188 GLI subjects)

Reports per-model: Loss, Accuracy, F1, AUC, Sensitivity, Specificity,
                   Confusion Matrix, and per-class breakdown.
Saves full report to results/evaluation_report.json
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_VAL_MANIFEST, BRATS2023_TEST_MANIFEST,
    MODELS_DIR, RESULTS_DIR,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model

RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINTS = {
    "resnet18_medicalnet": MODELS_DIR / "resnet18_medicalnet_lgg_hgg_3d.pt",
    "densenet121_3d":      MODELS_DIR / "densenet121_3d_lgg_hgg_3d.pt",
    "swin_transformer_3d": MODELS_DIR / "swin_transformer_3d_lgg_hgg_3d.pt",
}
LABEL_NAMES = {0: "LGG", 1: "HGG"}


# ── Evaluate one split ────────────────────────────────────────────────────────

def evaluate_split(model, loader, device, criterion):
    model.eval()
    losses, y_true, y_prob = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(imgs)
                loss   = criterion(logits, labels)
            losses.append(loss.item())
            y_true.append(labels.cpu().numpy())
            y_prob.append(torch.softmax(logits, dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    # Sensitivity (recall for LGG=0) and Specificity (recall for HGG=1)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # HGG recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # LGG recall

    lgg_n = int((y_true == 0).sum())
    hgg_n = int((y_true == 1).sum())
    lgg_correct = int(((y_pred == 0) & (y_true == 0)).sum())
    hgg_correct = int(((y_pred == 1) & (y_true == 1)).sum())

    return {
        "n_subjects":    len(y_true),
        "loss":          float(np.mean(losses)),
        "accuracy":      float(acc),
        "f1_weighted":   float(f1),
        "auc":           float(auc),
        "sensitivity":   float(sensitivity),   # HGG recall
        "specificity":   float(specificity),   # LGG recall
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "LGG": {"n": lgg_n, "correct": lgg_correct,
                    "recall": round(lgg_correct/lgg_n, 4) if lgg_n else 0},
            "HGG": {"n": hgg_n, "correct": hgg_correct,
                    "recall": round(hgg_correct/hgg_n, 4) if hgg_n else 0},
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load manifests
    def get_loader(manifest_path):
        df = load_manifest(manifest_path, label_col="label_grade")
        df = df[df["task"] == "GLI"].reset_index(drop=True)
        ds = BraTS2023Dataset(df, transforms=get_val_transforms(), label_col="label_grade")
        return DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True), len(df)

    val_loader,  n_val  = get_loader(BRATS2023_VAL_MANIFEST)
    test_loader, n_test = get_loader(BRATS2023_TEST_MANIFEST)
    print(f"Val  subjects: {n_val}")
    print(f"Test subjects: {n_test}\n")

    all_results = {}

    for model_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cw   = torch.tensor(ckpt["class_weights"], dtype=torch.float32, device=device)
        model = get_model(model_name, num_classes=2)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)

        print(f"  Epochs trained : {len(ckpt.get('history', []))}")
        print(f"  Best val acc   : {ckpt.get('best_val_accuracy', '?'):.4f}\n")

        print("  [VAL] Running...")
        val_r  = evaluate_split(model, val_loader,  device, criterion)
        print("  [TEST] Running...")
        test_r = evaluate_split(model, test_loader, device, criterion)

        # Print results
        for split_name, r in [("VAL ", val_r), ("TEST", test_r)]:
            print(f"\n  {split_name} | n={r['n_subjects']}")
            print(f"    Loss        : {r['loss']:.4f}")
            print(f"    Accuracy    : {r['accuracy']:.4f}  ({r['accuracy']*100:.2f}%)")
            print(f"    F1 (wtd)    : {r['f1_weighted']:.4f}")
            print(f"    AUC         : {r['auc']:.4f}")
            print(f"    Sensitivity : {r['sensitivity']:.4f}  (HGG recall)")
            print(f"    Specificity : {r['specificity']:.4f}  (LGG recall)")
            print(f"    LGG  {r['per_class']['LGG']['correct']:>3}/{r['per_class']['LGG']['n']:>3}"
                  f"  recall={r['per_class']['LGG']['recall']:.3f}")
            print(f"    HGG  {r['per_class']['HGG']['correct']:>3}/{r['per_class']['HGG']['n']:>3}"
                  f"  recall={r['per_class']['HGG']['recall']:.3f}")

        all_results[model_name] = {
            "epochs_trained":    len(ckpt.get("history", [])),
            "best_val_accuracy": ckpt.get("best_val_accuracy"),
            "val":  val_r,
            "test": test_r,
        }
        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'='*72}")
    print(f"  FINAL EVALUATION SUMMARY")
    print(f"{'='*72}")
    print(f"  {'Model':<26} {'Split':<5} {'Acc':>7} {'AUC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}")
    print(f"  {'-'*67}")
    for mname, r in all_results.items():
        for split in ["val", "test"]:
            s = r[split]
            print(f"  {mname:<26} {split.upper():<5} {s['accuracy']:>7.4f} "
                  f"{s['auc']:>7.4f} {s['f1_weighted']:>7.4f} "
                  f"{s['sensitivity']:>7.4f} {s['specificity']:>7.4f}")
    print(f"{'='*72}")

    out = RESULTS_DIR / "evaluation_report.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Full report saved: {out}")


if __name__ == "__main__":
    main()
