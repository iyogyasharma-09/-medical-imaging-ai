"""
evaluate_locked_test.py
------------------------
Phase 1: Locked Test Evaluation Engine for BraTS 2023 GLI 3D Grading Pipeline.

Evaluates all 9 canonical checkpoints:
  3 Architectures: ResNet18 MedicalNet, DenseNet121 3D, SwinUNETR 3D
  3 Protocols: A (Unweighted), B (Canonical IF Weights), C (Adjusted Minority Emphasis)

Safety & Freeze Rules:
  - Inference only (`model.eval()`, `torch.no_grad()`)
  - No parameter updates, no optimizer step
  - No retraining, no threshold tuning, no calibration fitting on test set
  - Isolated outputs under results/locked_test_evaluation/
"""

import sys
import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import BRATS2023_TEST_MANIFEST, BRATS2023_VAL_MANIFEST, RESULTS_DIR
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_val_transforms
from train_brats2023 import get_model

# ---------------------------------------------------------------------------
# Directories setup
# ---------------------------------------------------------------------------
OUTPUT_DIR = RESULTS_DIR / "locked_test_evaluation"
PRED_DIR   = OUTPUT_DIR / "predictions"
METRIC_DIR = OUTPUT_DIR / "metrics"
CM_DIR     = OUTPUT_DIR / "confusion_matrices"
LOG_DIR    = OUTPUT_DIR / "logs"

for d in [OUTPUT_DIR, PRED_DIR, METRIC_DIR, CM_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ARCHIVE_DIR = RESULTS_DIR / "protocol_archives"

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


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def audit_archive_hashes() -> dict:
    hashes = {}
    for proto in PROTOCOLS:
        proto_path = ARCHIVE_DIR / proto
        for f in proto_path.iterdir():
            if f.is_file():
                rel_path = f"{proto}/{f.name}"
                hashes[rel_path] = compute_sha256(f)
    return hashes


def compute_uncertainty(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Predictive entropy in bits: - sum(p * log2(p))"""
    p0_c = np.clip(p0, 1e-12, 1.0)
    p1_c = np.clip(p1, 1e-12, 1.0)
    entropy = - (p0_c * np.log2(p0_c) + p1_c * np.log2(p1_c))
    return entropy


def run_inference(model, loader, device) -> tuple[pd.DataFrame, dict]:
    model.eval()
    subject_ids, y_true, y_prob_0, y_prob_1 = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Inference"):
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            s_ids = batch.get("subject_id", [f"sub_{i}" for i in range(len(imgs))])

            with torch.amp.autocast(device_type=device.type, enabled=True):
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)

            subject_ids.extend(s_ids)
            y_true.extend(labels.cpu().numpy())
            probs_np = probs.cpu().numpy()
            y_prob_0.extend(probs_np[:, 0])
            y_prob_1.extend(probs_np[:, 1])

    y_true   = np.array(y_true)
    y_prob_0 = np.array(y_prob_0)
    y_prob_1 = np.array(y_prob_1)
    y_pred   = (y_prob_1 >= 0.5).astype(int)

    confidence  = np.maximum(y_prob_0, y_prob_1)
    uncertainty = compute_uncertainty(y_prob_0, y_prob_1)

    df_preds = pd.DataFrame({
        "subject_id":      subject_ids,
        "true_label":      y_true,
        "predicted_label": y_pred,
        "prob_lgg":        y_prob_0,
        "prob_hgg":        y_prob_1,
        "confidence":       confidence,
        "uncertainty":      uncertainty,
    })

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
    precision  = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    metrics = {
        "n_subjects":    len(y_true),
        "accuracy":      acc,
        "f1_macro":      f1_m,
        "f1_weighted":   f1_w,
        "auc":           auc,
        "lgg_recall":    lgg_recall,
        "hgg_recall":    hgg_recall,
        "precision":     precision,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "matrix": cm.tolist(),
        },
    }

    return df_preds, metrics


def main():
    start_time = time.time()

    log_file = LOG_DIR / "evaluation_execution.log"
    log_fp = open(log_file, "w", encoding="utf-8")

    def log(msg: str):
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', errors='replace').decode('ascii'))
        log_fp.write(msg + "\n")
        log_fp.flush()

    log("=" * 80)
    log("DAY 5 — PHASE 1: LOCKED TEST EVALUATION EXECUTION")
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        log(f"GPU Name: {torch.cuda.get_device_name(0)}")
        log(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    log(f"PyTorch Version: {torch.__version__}")
    log(f"Python Version: {sys.version.split()[0]}")

    log("\n[STEP 0] Running Pre-Flight Archive SHA-256 Hash Computation...")
    pre_hashes = audit_archive_hashes()
    log(f"Recorded SHA-256 hashes for {len(pre_hashes)} files in protocol archives.")

    # Load test & val manifests
    df_test_raw = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")
    df_test = df_test_raw[df_test_raw["task"] == "GLI"].reset_index(drop=True)
    ds_test = BraTS2023Dataset(df_test, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    test_loader = DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    df_val_raw = load_manifest(BRATS2023_VAL_MANIFEST, label_col="label_grade")
    df_val = df_val_raw[df_val_raw["task"] == "GLI"].reset_index(drop=True)
    ds_val = BraTS2023Dataset(df_val, transforms=get_val_transforms(), label_col="label_grade", return_meta=True)
    val_loader = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    log(f"\nLoaded Datasets:")
    log(f"  Test subjects (GLI): {len(df_test)} (LGG={sum(df_test['label_grade']==0)}, HGG={sum(df_test['label_grade']==1)})")
    log(f"  Val  subjects (GLI): {len(df_val)} (LGG={sum(df_val['label_grade']==0)}, HGG={sum(df_val['label_grade']==1)})")

    results_table = []
    ablation_records = []
    generalization_records = []
    all_eval_data = {}

    log("\n" + "=" * 80)
    log("[STEP 1-6] STARTING LOCKED TEST EVALUATION ACROSS ALL 9 CHECKPOINTS")
    log("=" * 80)

    for proto in PROTOCOLS:
        proto_short = proto.split("_")[1]  # 'A', 'B', or 'C'
        proto_dir = ARCHIVE_DIR / proto

        for model_key, ckpt_filename in MODELS.items():
            disp_name = MODEL_DISPLAY_NAMES[model_key]
            file_prefix = FILE_PREFIX_MAP[model_key]

            log(f"\n>>> Evaluating: {disp_name} | Protocol {proto_short}")
            ckpt_path = proto_dir / ckpt_filename

            if not ckpt_path.exists():
                raise FileNotFoundError(f"CRITICAL: Checkpoint missing: {ckpt_path}")

            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            cw_raw = ckpt.get("class_weights", None)
            best_val_macro_f1 = ckpt.get("best_val_macro_f1", None)
            best_val_lgg_rec  = ckpt.get("best_val_lgg_recall", None)

            log(f"    Checkpoint size: {ckpt_path.stat().st_size} bytes")
            log(f"    Protocol in metadata: {ckpt.get('training_protocol', 'N/A')}")
            log(f"    Class weights in metadata: {cw_raw}")
            log(f"    Stored best val Macro-F1: {best_val_macro_f1}")
            log(f"    Stored best val LGG Recall: {best_val_lgg_rec}")

            # Instantiate model
            model = get_model(model_key, num_classes=2)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(device)

            # Test inference
            log("    Running Test Inference...")
            df_preds, test_m = run_inference(model, test_loader, device)

            # Val inference (for exact Val vs Test comparison)
            log("    Running Val Inference...")
            _, val_m = run_inference(model, val_loader, device)

            # Save patient predictions CSV
            pred_filename = f"{file_prefix}_protocol{proto_short}_test_predictions.csv"
            pred_filepath = PRED_DIR / pred_filename
            df_preds.to_csv(pred_filepath, index=False)
            log(f"    Saved patient predictions: {pred_filepath.name} ({len(df_preds)} rows)")

            # Save CM JSON
            cm_filename = f"{file_prefix}_protocol{proto_short}_cm.json"
            cm_filepath = CM_DIR / cm_filename
            cm_data = {
                "model": disp_name,
                "protocol": proto_short,
                "confusion_matrix": test_m["confusion_matrix"],
                "lgg_recall": test_m["lgg_recall"],
                "hgg_recall": test_m["hgg_recall"],
                "accuracy": test_m["accuracy"],
                "f1_macro": test_m["f1_macro"],
                "auc": test_m["auc"],
            }
            cm_filepath.write_text(json.dumps(cm_data, indent=2))

            # Store result row
            results_table.append({
                "model_key":     model_key,
                "Model":         disp_name,
                "Protocol":      proto_short,
                "Test Accuracy": round(test_m["accuracy"], 4),
                "Test Macro-F1": round(test_m["f1_macro"], 4),
                "Test LGG Recall": round(test_m["lgg_recall"], 4),
                "Test AUC":      round(test_m["auc"], 4),
                "Test HGG Recall": round(test_m["hgg_recall"], 4),
                "Test Precision":  round(test_m["precision"], 4),
                "Test Weighted-F1": round(test_m["f1_weighted"], 4),
            })

            generalization_records.append({
                "Model":           disp_name,
                "Protocol":        proto_short,
                "Val Macro-F1":    round(best_val_macro_f1 if best_val_macro_f1 is not None else val_m["f1_macro"], 4),
                "Test Macro-F1":   round(test_m["f1_macro"], 4),
                "Val LGG Recall":  round(best_val_lgg_rec if best_val_lgg_rec is not None else val_m["lgg_recall"], 4),
                "Test LGG Recall": round(test_m["lgg_recall"], 4),
                "Val AUC":         round(val_m["auc"], 4),
                "Test AUC":        round(test_m["auc"], 4),
            })

            all_eval_data[f"{model_key}_{proto}"] = {
                "model": disp_name,
                "protocol": proto_short,
                "test_metrics": test_m,
                "val_metrics": val_m,
                "stored_best_val_macro_f1": best_val_macro_f1,
                "stored_best_val_lgg_recall": best_val_lgg_rec,
            }

            log(f"    RESULTS -> Acc: {test_m['accuracy']:.4f} | Macro-F1: {test_m['f1_macro']:.4f} | LGG Recall: {test_m['lgg_recall']:.4f} | AUC: {test_m['auc']:.4f}")

            del model
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Step 5: Save Final 9-Model Table
    # ---------------------------------------------------------------------------
    df_results = pd.DataFrame(results_table)
    summary_cols = ["Model", "Protocol", "Test Accuracy", "Test Macro-F1", "Test LGG Recall", "Test AUC"]
    df_summary = df_results[summary_cols]

    csv_summary_path = METRIC_DIR / "locked_test_metrics_summary.csv"
    json_summary_path = METRIC_DIR / "locked_test_metrics_summary.json"
    df_summary.to_csv(csv_summary_path, index=False)
    json_summary_path.write_text(json.dumps(df_results.to_dict(orient="records"), indent=2))

    log("\n" + "=" * 80)
    log("[STEP 5] FINAL 9-MODEL LOCKED TEST METRICS TABLE")
    log("=" * 80)
    log(df_summary.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Step 7: Ablation Study (Protocol A vs Protocol C)
    # ---------------------------------------------------------------------------
    for mkey in MODELS.keys():
        disp_name = MODEL_DISPLAY_NAMES[mkey]
        row_A = [r for r in results_table if r["model_key"] == mkey and r["Protocol"] == "A"][0]
        row_C = [r for r in results_table if r["model_key"] == mkey and r["Protocol"] == "C"][0]

        f1_A = row_A["Test Macro-F1"]
        f1_C = row_C["Test Macro-F1"]
        delta_f1 = round(f1_C - f1_A, 4)

        rec_A = row_A["Test LGG Recall"]
        rec_C = row_C["Test LGG Recall"]
        delta_rec = round(rec_C - rec_A, 4)

        auc_A = row_A["Test AUC"]
        auc_C = row_C["Test AUC"]
        delta_auc = round(auc_C - auc_A, 4)

        ablation_records.append({
            "Model": disp_name,
            "Protocol A Macro-F1": f1_A,
            "Protocol C Macro-F1": f1_C,
            "Delta Macro-F1": delta_f1,
            "Protocol A LGG Recall": rec_A,
            "Protocol C LGG Recall": rec_C,
            "Delta LGG Recall": delta_rec,
            "Protocol A AUC": auc_A,
            "Protocol C AUC": auc_C,
            "Delta AUC": delta_auc,
        })

    df_ablation = pd.DataFrame(ablation_records)
    ablation_csv = METRIC_DIR / "ablation_protocol_A_vs_C.csv"
    ablation_json = METRIC_DIR / "ablation_protocol_A_vs_C.json"
    df_ablation.to_csv(ablation_csv, index=False)
    ablation_json.write_text(json.dumps(ablation_records, indent=2))

    log("\n" + "=" * 80)
    log("[STEP 7] ABLATION STUDY TABLE (Protocol A vs Protocol C)")
    log("=" * 80)
    log(df_ablation.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Step 8: Validation vs Test Generalization Gap
    # ---------------------------------------------------------------------------
    df_gen = pd.DataFrame(generalization_records)
    gen_csv = METRIC_DIR / "validation_vs_test_generalization.csv"
    gen_json = METRIC_DIR / "validation_vs_test_generalization.json"
    df_gen.to_csv(gen_csv, index=False)
    gen_json.write_text(json.dumps(generalization_records, indent=2))

    log("\n" + "=" * 80)
    log("[STEP 8] VALIDATION VS TEST GENERALIZATION GAP TABLE")
    log("=" * 80)
    log(df_gen.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Step 11: Safety Check After Evaluation (Integrity Re-Verification)
    # ---------------------------------------------------------------------------
    log("\n" + "=" * 80)
    log("[STEP 11] POST-EVALUATION INTEGRITY SAFETY CHECK")
    log("=" * 80)
    post_hashes = audit_archive_hashes()

    mismatches = 0
    for rel_path, pre_hash in pre_hashes.items():
        post_hash = post_hashes.get(rel_path, None)
        if post_hash != pre_hash:
            log(f"  [CRITICAL ERROR] Hash mismatch on file: {rel_path}!")
            mismatches += 1

    if mismatches == 0:
        log(f"  [SUCCESS] All {len(pre_hashes)} archived protocol files are 100% UNCHANGED!")
    else:
        log(f"  [FAILURE] {mismatches} file(s) modified! Integrity violated!")

    total_time = time.time() - start_time
    log(f"\nTotal Evaluation Runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    log("=" * 80)
    log_fp.close()
    print(f"\nExecution log written to: {log_file}")


if __name__ == "__main__":
    main()
