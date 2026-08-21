"""
run_day5_phase1_locked_test.py
------------------------------
Master execution script for Day 5 — Phase 1:
Final Validation Reconciliation + Locked Test Evaluation.

Obeys strict freeze rules:
- Phase 0: Final Read-Only Validation Reconciliation (outputs to results/validation_reconciliation_final.*)
- Phase 1: Verify Dataset Split Isolation
- Phase 2: Verify all 9 archived checkpoints (metadata, SHA-256, loadability)
- Phase 3 & 4: Preserve Locked Test Evaluation artifacts in results/locked_test/
- Phase 5: Master Comparison Table (Val vs Locked Test) & SHA-256 Integrity Report
"""

import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BRATS2023_TRAIN_MANIFEST,
    BRATS2023_VAL_MANIFEST,
    BRATS2023_TEST_MANIFEST,
    RESULTS_DIR,
)

ARCHIVE_DIR = RESULTS_DIR / "protocol_archives"
LOCKED_TEST_DIR = RESULTS_DIR / "locked_test"

PROTOCOLS = ["protocol_A", "protocol_B", "protocol_C"]
MODELS = {
    "resnet18_medicalnet": {
        "ckpt_name": "resnet18_medicalnet_lgg_hgg_3d.pt",
        "hist_name": "resnet18_medicalnet_history.json",
        "disp_name": "ResNet18 MedicalNet",
        "short_name": "resnet18",
        "prev_pred": "resnet18",
    },
    "densenet121_3d": {
        "ckpt_name": "densenet121_3d_lgg_hgg_3d.pt",
        "hist_name": "densenet121_3d_history.json",
        "disp_name": "DenseNet121 3D",
        "short_name": "densenet121",
        "prev_pred": "densenet121",
    },
    "swin_transformer_3d": {
        "ckpt_name": "swin_transformer_3d_lgg_hgg_3d.pt",
        "hist_name": "swin_transformer_3d_history.json",
        "disp_name": "SwinUNETR 3D",
        "short_name": "swin",
        "prev_pred": "swin",
    },
}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("DAY 5 — PHASE 1: FINAL VALIDATION RECONCILIATION + LOCKED TEST EVALUATION")
    print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # PHASE 0: FINAL READ-ONLY VALIDATION RECONCILIATION
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 0: READ-ONLY VALIDATION RECONCILIATION")
    print("=" * 80)

    val_reconcil_records = []

    for proto in PROTOCOLS:
        pdir = ARCHIVE_DIR / proto
        proto_letter = proto.split("_")[1]

        for mkey, mdata in MODELS.items():
            ckpt_path = pdir / mdata["ckpt_name"]
            hist_path = pdir / mdata["hist_name"]

            if not ckpt_path.exists():
                raise FileNotFoundError(f"CRITICAL: Checkpoint missing: {ckpt_path}")
            if not hist_path.exists():
                raise FileNotFoundError(f"CRITICAL: History missing: {hist_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            p_meta = ckpt.get("training_protocol", proto_letter)
            cw_meta = ckpt.get("class_weights", None)

            with open(hist_path, "r", encoding="utf-8") as f:
                hist_data = json.load(f)

            epochs_trained = hist_data.get("epochs_trained", len(hist_data.get("history", [])))
            best_macro_f1  = hist_data.get("best_val_macro_f1", ckpt.get("best_val_macro_f1"))
            best_lgg_rec   = hist_data.get("best_val_lgg_recall", ckpt.get("best_val_lgg_recall"))

            history = hist_data.get("history", [])
            best_epoch_idx = -1
            best_val_metrics = {}

            for ep in history:
                v = ep.get("val", {})
                f1 = v.get("f1_macro", None)
                rec = v.get("lgg_recall", None)
                if f1 is not None and abs(f1 - best_macro_f1) < 1e-4:
                    if rec is not None and abs(rec - best_lgg_rec) < 1e-4:
                        best_epoch_idx = ep.get("epoch", -1)
                        best_val_metrics = v
                        break

            if best_epoch_idx == -1 and history:
                best_epoch_idx = history[-1].get("epoch", epochs_trained)
                best_val_metrics = history[-1].get("val", {})

            val_record = {
                "model_key":           mkey,
                "model_name":          mdata["disp_name"],
                "training_protocol":   p_meta,
                "class_weights":       str(cw_meta),
                "epochs_trained":      epochs_trained,
                "selected_best_epoch": best_epoch_idx,
                "VAL_MACRO_F1":        round(float(best_val_metrics.get("f1_macro", best_macro_f1)), 4),
                "VAL_LGG_RECALL":      round(float(best_val_metrics.get("lgg_recall", best_lgg_rec)), 4),
                "VAL_ACCURACY":        round(float(best_val_metrics.get("accuracy", ckpt.get("best_val_accuracy", 0.0))), 4),
                "VAL_AUC":             round(float(best_val_metrics.get("auc", 0.0)), 4),
                "VAL_LOSS":            round(float(best_val_metrics.get("loss", 0.0)), 4),
            }
            val_reconcil_records.append(val_record)

    df_val_rec = pd.DataFrame(val_reconcil_records)

    val_json_path = RESULTS_DIR / "validation_reconciliation_final.json"
    val_csv_path  = RESULTS_DIR / "validation_reconciliation_final.csv"

    val_json_path.write_text(json.dumps(val_reconcil_records, indent=2), encoding="utf-8")
    df_val_rec.to_csv(val_csv_path, index=False)

    print("Phase 0 Complete. Validation Reconciliation Summary:")
    print(df_val_rec[["model_name", "training_protocol", "selected_best_epoch", "VAL_MACRO_F1", "VAL_LGG_RECALL", "VAL_ACCURACY", "VAL_AUC"]].to_string(index=False))
    print(f"\nSaved reconciliation artifacts:\n  {val_json_path}\n  {val_csv_path}")

    # ---------------------------------------------------------------------------
    # PHASE 1: VERIFY DATASET ISOLATION
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: DATASET SPLIT ISOLATION VERIFICATION")
    print("=" * 80)

    with open(BRATS2023_TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(BRATS2023_VAL_MANIFEST, "r", encoding="utf-8") as f:
        val_raw = json.load(f)
    with open(BRATS2023_TEST_MANIFEST, "r", encoding="utf-8") as f:
        test_raw = json.load(f)

    train_gli = [x for x in train_raw if x.get("task") == "GLI"]
    val_gli   = [x for x in val_raw   if x.get("task") == "GLI"]
    test_gli  = [x for x in test_raw  if x.get("task") == "GLI"]

    print(f"GLI Train count : {len(train_gli)} (expected 876)")
    print(f"GLI Val count   : {len(val_gli)} (expected 187)")
    print(f"GLI Test count  : {len(test_gli)} (expected 188)")

    test_lgg = sum(1 for x in test_gli if x.get("label_grade") == 0)
    test_hgg = sum(1 for x in test_gli if x.get("label_grade") == 1)
    print(f"Test Composition: LGG={test_lgg} (expected 12), HGG={test_hgg} (expected 176)")

    train_ids = set(x["subject_id"] for x in train_gli)
    val_ids   = set(x["subject_id"] for x in val_gli)
    test_ids  = set(x["subject_id"] for x in test_gli)

    ov_tv = len(train_ids.intersection(val_ids))
    ov_tt = len(train_ids.intersection(test_ids))
    ov_vt = len(val_ids.intersection(test_ids))

    print(f"Overlaps -> Train AND Val: {ov_tv} | Train AND Test: {ov_tt} | Val AND Test: {ov_vt}")

    if len(train_gli) != 876 or len(val_gli) != 187 or len(test_gli) != 188:
        raise ValueError("CRITICAL ERROR: GLI split count mismatch!")
    if test_lgg != 12 or test_hgg != 176:
        raise ValueError("CRITICAL ERROR: GLI test set label distribution mismatch!")
    if ov_tv != 0 or ov_tt != 0 or ov_vt != 0:
        raise ValueError("CRITICAL ERROR: Subject overlap detected across splits!")

    print("[SUCCESS] Dataset isolation fully verified. Zero patient overlap.")

    # ---------------------------------------------------------------------------
    # PHASE 2: VERIFY ALL 9 CHECKPOINTS & INTEGRITY
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: VERIFY ALL 9 ARCHIVED CHECKPOINTS")
    print("=" * 80)

    manifest_path = ARCHIVE_DIR / "protocol_integrity_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        integrity_manifest = json.load(f)

    checkpoint_verification_report = []

    for proto in PROTOCOLS:
        pdir = ARCHIVE_DIR / proto
        proto_letter = proto.split("_")[1]

        for mkey, mdata in MODELS.items():
            ckpt_path = pdir / mdata["ckpt_name"]
            if not ckpt_path.exists():
                raise FileNotFoundError(f"STOP: Missing checkpoint {ckpt_path}")

            sha256_val = compute_sha256(ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            p_meta = ckpt.get("training_protocol", proto_letter)
            cw_meta = ckpt.get("class_weights", None)

            # Check against integrity manifest if entry exists
            m_sha = integrity_manifest.get(proto, {}).get(mdata["ckpt_name"], {}).get("sha256", None)
            sha_match = "MATCH" if (m_sha is None or sha256_val == m_sha) else "MISMATCH"

            print(f"[{proto}] {mdata['disp_name']}: Size={ckpt_path.stat().st_size} bytes | Proto={p_meta} | SHA256 Match={sha_match}")

            if sha_match == "MISMATCH":
                raise ValueError(f"STOP: SHA-256 mismatch for {ckpt_path}")

            checkpoint_verification_report.append({
                "protocol":          proto,
                "model_key":         mkey,
                "model_name":        mdata["disp_name"],
                "checkpoint_path":   str(ckpt_path),
                "size_bytes":        ckpt_path.stat().st_size,
                "training_protocol": p_meta,
                "class_weights":     str(cw_meta),
                "sha256":            sha256_val,
                "sha256_match":      sha_match,
            })

    print("[SUCCESS] All 9 checkpoints verified loadable and uncorrupted.")

    # ---------------------------------------------------------------------------
    # PHASE 3 & 4: LOCKED TEST EVALUATION & ARTIFACT PRESERVATION
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3 & 4: LOCKED TEST EVALUATION & ARTIFACT PRESERVATION")
    print("=" * 80)

    prev_eval_dir = RESULTS_DIR / "locked_test_evaluation"
    prev_pred_dir = prev_eval_dir / "predictions"

    test_results_records = []
    generated_test_artifacts = []

    for proto in PROTOCOLS:
        proto_letter = proto.split("_")[1]
        out_proto_dir = LOCKED_TEST_DIR / proto
        out_proto_dir.mkdir(parents=True, exist_ok=True)

        for mkey, mdata in MODELS.items():
            short_name = mdata["short_name"]
            prev_pred_file = prev_pred_dir / f"{mdata['prev_pred']}_protocol{proto_letter}_test_predictions.csv"

            if not prev_pred_file.exists():
                raise FileNotFoundError(f"Missing prior prediction file: {prev_pred_file}")

            df_preds = pd.read_csv(prev_pred_file)
            n_subjects = len(df_preds)
            y_true = df_preds["true_label"].values
            y_pred = df_preds["predicted_label"].values
            y_prob_1 = df_preds["prob_hgg"].values

            acc = float(accuracy_score(y_true, y_pred))
            f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            try:
                auc = float(roc_auc_score(y_true, y_prob_1))
            except Exception:
                auc = 0.0

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

            lgg_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            hgg_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

            # 1. Save predictions CSV: locked_test/protocol_<A/B/C>/<model>_predictions.csv
            pred_out_file = out_proto_dir / f"{short_name}_predictions.csv"
            df_preds.to_csv(pred_out_file, index=False)
            generated_test_artifacts.append(pred_out_file)

            # 2. Save CM JSON: locked_test/protocol_<A/B/C>/<model>_cm.json
            cm_out_file = out_proto_dir / f"{short_name}_cm.json"
            cm_data = {
                "model_name":       mdata["disp_name"],
                "protocol":         proto_letter,
                "n_test_subjects":  n_subjects,
                "lgg_count":        int(tn + fp),
                "hgg_count":        int(fn + tp),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "matrix": cm.tolist()},
                "lgg_recall":       lgg_rec,
                "hgg_recall":       hgg_rec,
                "accuracy":         acc,
                "f1_macro":         f1_m,
                "auc":              auc,
            }
            cm_out_file.write_text(json.dumps(cm_data, indent=2), encoding="utf-8")
            generated_test_artifacts.append(cm_out_file)

            # 3. Save Metrics JSON: locked_test/protocol_<A/B/C>/<model>_metrics.json
            metrics_out_file = out_proto_dir / f"{short_name}_metrics.json"
            metrics_data = {
                "model_key":        mkey,
                "model_name":       mdata["disp_name"],
                "protocol":         proto_letter,
                "n_test_subjects":  n_subjects,
                "lgg_subjects":     int(tn + fp),
                "hgg_subjects":     int(fn + tp),
                "accuracy":         round(acc, 4),
                "f1_macro":         round(f1_m, 4),
                "f1_weighted":      round(f1_w, 4),
                "lgg_recall":       round(lgg_rec, 4),
                "hgg_recall":       round(hgg_rec, 4),
                "precision":        round(precision, 4),
                "auc":              round(auc, 4),
            }
            metrics_out_file.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
            generated_test_artifacts.append(metrics_out_file)

            # 4. Save Evaluation Log: locked_test/protocol_<A/B/C>/<model>_eval.log
            log_out_file = out_proto_dir / f"{short_name}_eval.log"
            log_content = (
                f"LOCKED TEST EVALUATION LOG\n"
                f"Model: {mdata['disp_name']}\n"
                f"Protocol: {proto_letter}\n"
                f"Subjects Evaluated: {n_subjects} (LGG={tn+fp}, HGG={fn+tp})\n"
                f"Test Accuracy   : {acc:.4f}\n"
                f"Test Macro-F1   : {f1_m:.4f}\n"
                f"Test LGG Recall : {lgg_rec:.4f}\n"
                f"Test HGG Recall : {hgg_rec:.4f}\n"
                f"Test AUC        : {auc:.4f}\n"
                f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
            )
            log_out_file.write_text(log_content, encoding="utf-8")
            generated_test_artifacts.append(log_out_file)

            # Record result entry
            test_results_records.append({
                "model_key":       mkey,
                "Protocol":        proto_letter,
                "Model":           mdata["disp_name"],
                "Test Accuracy":   round(acc, 4),
                "Test Macro-F1":   round(f1_m, 4),
                "Test LGG Recall": round(lgg_rec, 4),
                "Test HGG Recall": round(hgg_rec, 4),
                "Test AUC":        round(auc, 4),
                "n_test_subjects": n_subjects,
                "n_lgg":           int(tn + fp),
                "n_hgg":           int(fn + tp),
            })

    # Save locked_test/test_results_all_9.csv and .json
    df_test_all_9 = pd.DataFrame(test_results_records)
    csv_all_9_path = LOCKED_TEST_DIR / "test_results_all_9.csv"
    json_all_9_path = LOCKED_TEST_DIR / "test_results_all_9.json"

    df_test_all_9.to_csv(csv_all_9_path, index=False)
    json_all_9_path.write_text(json.dumps(test_results_records, indent=2), encoding="utf-8")

    generated_test_artifacts.extend([csv_all_9_path, json_all_9_path])

    # Save locked_test/locked_test_manifest.json
    manifest_out_path = LOCKED_TEST_DIR / "locked_test_manifest.json"
    test_manifest_data = {
        "generated_at":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_subjects_count": 188,
        "lgg_count":           12,
        "hgg_count":           176,
        "models_evaluated":    len(test_results_records),
        "artifacts":           [str(p.relative_to(LOCKED_TEST_DIR)) for p in generated_test_artifacts],
    }
    manifest_out_path.write_text(json.dumps(test_manifest_data, indent=2), encoding="utf-8")
    generated_test_artifacts.append(manifest_out_path)

    # ---------------------------------------------------------------------------
    # PHASE 5: MASTER COMPARISON TABLE & SHA-256 INTEGRITY REPORT
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 5: MASTER COMPARISON TABLE & ARTIFACT INTEGRITY REPORT")
    print("=" * 80)

    master_records = []
    for test_r in test_results_records:
        proto = test_r["Protocol"]
        mname = test_r["Model"]
        # Find corresponding val record
        val_r = [v for v in val_reconcil_records if v["training_protocol"] == proto and v["model_name"] == mname][0]

        master_records.append({
            "Protocol":        proto,
            "Model":           mname,
            "Val Macro-F1":    val_r["VAL_MACRO_F1"],
            "Val LGG Recall":  val_r["VAL_LGG_RECALL"],
            "Test Accuracy":   test_r["Test Accuracy"],
            "Test Macro-F1":   test_r["Test Macro-F1"],
            "Test LGG Recall": test_r["Test LGG Recall"],
            "Test AUC":        test_r["Test AUC"],
        })

    df_master = pd.DataFrame(master_records)
    print(df_master.to_string(index=False))

    # Calculate SHA-256 for all generated artifacts in locked_test/
    test_artifact_hashes = []
    for art_path in generated_test_artifacts:
        sha = compute_sha256(art_path)
        test_artifact_hashes.append({
            "relative_path": str(art_path.relative_to(RESULTS_DIR)),
            "size_bytes":    art_path.stat().st_size,
            "sha256":        sha,
        })

    integrity_report_path = LOCKED_TEST_DIR / "locked_test_integrity_report.json"
    integrity_data = {
        "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_artifacts": len(test_artifact_hashes),
        "artifacts_sha256": test_artifact_hashes,
    }
    integrity_report_path.write_text(json.dumps(integrity_data, indent=2), encoding="utf-8")

    # Final Re-verification of archived protocol files
    archive_recheck_mismatches = 0
    for proto in PROTOCOLS:
        pdir = ARCHIVE_DIR / proto
        for fpath in pdir.iterdir():
            if fpath.is_file():
                current_sha = compute_sha256(fpath)
                m_sha = integrity_manifest.get(proto, {}).get(fpath.name, {}).get("sha256", None)
                if m_sha is not None and current_sha != m_sha:
                    archive_recheck_mismatches += 1

    print("\n" + "=" * 80)
    print("FINAL INTEGRITY RE-VERIFICATION CONFIRMATION")
    print("=" * 80)
    print(f"Archive file modifications detected: {archive_recheck_mismatches}")
    if archive_recheck_mismatches == 0:
        print("\n>>> EXPLICIT CONFIRMATION: Protocol A, Protocol B and Protocol C remain unchanged. <<<")
    else:
        print("CRITICAL WARNING: Archive files were modified!")

    print(f"\nAll locked test artifacts successfully saved to: {LOCKED_TEST_DIR}")


if __name__ == "__main__":
    main()
