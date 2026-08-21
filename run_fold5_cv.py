"""
run_fold5_cv.py
---------------
Day 6 — Phase 6: Execute ONLY Fold 5 of 5-Fold Cross-Validation for ResNet18 MedicalNet (Protocol A).

Safety & Integrity Rules:
- Execute ONLY Fold 5
- DO NOT modify Fold 1, Fold 2, Fold 3, or Fold 4
- DO NOT modify Protocol A/B/C archives
- DO NOT modify locked test predictions
- DO NOT modify dataset or cv_split_manifest.json
- Use exact configuration from run_phase6_cv.py
"""

import sys
import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    RESULTS_DIR,
    BRATS2023_TRAIN_MANIFEST,
    BRATS2023_VAL_MANIFEST,
    BRATS2023_TEST_MANIFEST,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED as SEED,
)

from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from train_brats2023 import (
    build_resnet18,
    get_train_transforms,
    get_val_transforms,
    EarlyStopping,
    get_training_device,
)

ARCHIVE_DIR     = RESULTS_DIR / "protocol_archives"
RELIABILITY_DIR = RESULTS_DIR / "reliability_analysis"
CV_DIR          = RELIABILITY_DIR / "phase6_cv"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def evaluate_val_set(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    targets = np.array(all_targets)
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    rec_lgg = float(recall_score(targets, preds, pos_label=0, zero_division=0))
    rec_hgg = float(recall_score(targets, preds, pos_label=1, zero_division=0))

    try:
        auc = float(roc_auc_score(targets, probs))
    except Exception:
        auc = 0.5

    return {
        "accuracy":   round(acc, 4),
        "f1_macro":   round(macro_f1, 4),
        "lgg_recall": round(rec_lgg, 4),
        "hgg_recall": round(rec_hgg, 4),
        "auc":        round(auc, 4),
    }


def main():
    print("=" * 80)
    print("DAY 6 — PHASE 6: EXECUTION OF FOLD 5 ONLY")
    print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # 1. Load Development Manifest (Train + Val GLI subjects)
    df_train_raw = load_manifest(BRATS2023_TRAIN_MANIFEST, label_col="label_grade")
    df_val_raw   = load_manifest(BRATS2023_VAL_MANIFEST, label_col="label_grade")
    df_test_raw  = load_manifest(BRATS2023_TEST_MANIFEST, label_col="label_grade")

    df_train_gli = df_train_raw[df_train_raw["task"] == "GLI"].copy().reset_index(drop=True)
    df_val_gli   = df_val_raw[df_val_raw["task"] == "GLI"].copy().reset_index(drop=True)
    df_test_gli  = df_test_raw[df_test_raw["task"] == "GLI"].copy().reset_index(drop=True)

    df_dev = pd.concat([df_train_gli, df_val_gli], ignore_index=True)

    test_subject_set = set(df_test_gli["subject_id"].unique())
    dev_subject_set  = set(df_dev["subject_id"].unique())
    overlap = dev_subject_set.intersection(test_subject_set)

    if len(overlap) > 0:
        raise ValueError("CRITICAL ERROR: Test leakage detected in development set!")

    # 2. Replicate Exact Stratified 5-Fold Split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Load existing cv_split_manifest to ensure 100% split match
    with open(CV_DIR / "cv_split_manifest.json", "r", encoding="utf-8") as f:
        cv_manifest = json.load(f)

    target_fold_idx = 5
    df_f_train = None
    df_f_val = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_dev, df_dev["label_grade"]), 1):
        if fold_idx == target_fold_idx:
            df_f_train = df_dev.iloc[train_idx].copy().reset_index(drop=True)
            df_f_val   = df_dev.iloc[val_idx].copy().reset_index(drop=True)
            break

    if df_f_train is None or df_f_val is None:
        raise RuntimeError("Failed to extract Fold 5 split!")

    # Verify against manifest
    manifest_val_subs = cv_manifest[f"fold_{target_fold_idx}"]["val_subjects"]
    actual_val_subs = df_f_val["subject_id"].tolist()
    if manifest_val_subs != actual_val_subs:
        raise ValueError("CRITICAL ERROR: Fold 5 validation subjects do not match cv_split_manifest.json!")

    print(f"Fold 5 Split Verified:")
    print(f"  Train subjects: {len(df_f_train)} (LGG={(df_f_train['label_grade']==0).sum()}, HGG={(df_f_train['label_grade']==1).sum()})")
    print(f"  Val subjects  : {len(df_f_val)} (LGG={(df_f_val['label_grade']==0).sum()}, HGG={(df_f_val['label_grade']==1).sum()})")

    # 3. Setup Fold 5 Training
    fold_dir = CV_DIR / f"fold_{target_fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    device = get_training_device()
    print(f"Training device: {device}")

    # Build datasets & loaders
    train_ds = BraTS2023Dataset(df_f_train, transforms=get_train_transforms(), label_col="label_grade")
    val_ds   = BraTS2023Dataset(df_f_val,   transforms=get_val_transforms(),   label_col="label_grade")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Seed for Fold 5: SEED + fold_idx (42 + 5 = 47)
    torch.manual_seed(SEED + target_fold_idx)
    np.random.seed(SEED + target_fold_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + target_fold_idx)

    model = build_resnet18(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    early_stopping = EarlyStopping(patience=5)

    max_epochs = 30
    best_val_loss = float("inf")
    best_val_metrics = {}
    epochs_trained = 0
    fold_history = []

    print(f"\n--- STARTING FOLD {target_fold_idx}/5 TRAINING ---")
    t_fold_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)

        train_loss = running_loss / len(train_ds)

        # Evaluate Val Set
        val_metrics = evaluate_val_set(model, val_loader, device)

        # Compute Val Loss
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_running_loss += loss.item() * len(labels)
        val_loss = val_running_loss / len(val_ds)
        val_metrics["val_loss"] = round(val_loss, 4)

        scheduler.step(val_loss)
        epochs_trained = epoch

        fold_history.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 4),
            "val_loss":    round(val_loss, 4),
            "accuracy":    val_metrics["accuracy"],
            "macro_f1":    val_metrics["f1_macro"],
            "lgg_recall":  val_metrics["lgg_recall"],
            "hgg_recall":  val_metrics["hgg_recall"],
            "auc":         val_metrics["auc"],
        })

        print(f"  [Fold {target_fold_idx} | Epoch {epoch:02d}/{max_epochs}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | acc: {val_metrics['accuracy']:.4f} | macro_f1: {val_metrics['f1_macro']:.4f} | lgg_recall: {val_metrics['lgg_recall']:.4f} | auc: {val_metrics['auc']:.4f}")

        stop_flag = early_stopping.step(val_metrics, model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if stop_flag:
            print(f"  [Fold {target_fold_idx}] Early stopping triggered at epoch {epoch}")
            break

    t_fold_end = time.time()
    elapsed_fold = round(t_fold_end - t_fold_start, 2)

    # Restore best model state
    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)

    best_epoch_num = early_stopping.best_epoch if hasattr(early_stopping, "best_epoch") else epochs_trained
    final_metrics = evaluate_val_set(model, val_loader, device)
    final_metrics["best_val_loss"] = round(best_val_loss, 4)

    # Save fold artifacts
    ckpt_out_path = fold_dir / "best_model.pth"
    hist_out_path = fold_dir / "history.json"

    torch.save(model.state_dict(), ckpt_out_path)
    with open(hist_out_path, "w", encoding="utf-8") as f:
        json.dump(fold_history, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[Fold {target_fold_idx} Complete]")
    print(f"  Epochs Trained : {epochs_trained}")
    print(f"  Best Epoch     : {best_epoch_num}")
    print(f"  Runtime        : {elapsed_fold:.2f}s ({elapsed_fold/60.0:.2f} mins)")
    print(f"  Val Accuracy   : {final_metrics['accuracy']:.4f}")
    print(f"  Val Macro-F1   : {final_metrics['f1_macro']:.4f}")
    print(f"  Val LGG Recall : {final_metrics['lgg_recall']:.4f}")
    print(f"  Val HGG Recall : {final_metrics['hgg_recall']:.4f}")
    print(f"  Val AUC        : {final_metrics['auc']:.4f}")
    print(f"  Best Val Loss  : {final_metrics['best_val_loss']:.4f}")
    print(f"  Saved Checkpoint: {ckpt_out_path} ({ckpt_out_path.stat().st_size} bytes)")
    print(f"  Saved History   : {hist_out_path} ({len(fold_history)} entries)")
    print("=" * 80)


if __name__ == "__main__":
    main()
