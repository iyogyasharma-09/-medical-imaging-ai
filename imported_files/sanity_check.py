"""
sanity_check.py
---------------
Three non-negotiable checks to run BEFORE touching training.

Check 1 — Label distribution   : counts + imbalance ratio
Check 2 — No data leakage      : zero overlap between splits
Check 3 — Overfit test         : model should reach ~zero loss on 5 samples
                                  within 20 epochs. If it doesn't, your
                                  data loader or model has a bug.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from typing import Dict


# ================================================================== #
# CHECK 1 — Label distribution                                       #
# ================================================================== #

def check_label_distribution(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    label_col: str = "label_grade",
) -> Dict[str, float]:
    """
    Print class counts and imbalance ratio for each split.
    Returns class weights for use in weighted loss.
    """
    from sklearn.utils.class_weight import compute_class_weight

    print("\n" + "="*55)
    print("CHECK 1 — Label Distribution")
    print("="*55)

    label_names = {0: "LGG", 1: "HGG"}

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = split[label_col].value_counts().sort_index()
        total  = len(split)
        parts  = [f"{label_names.get(k,k)}={v} ({100*v/total:.1f}%)" for k, v in counts.items()]
        print(f"  {name:5s} ({total:4d}) : {' | '.join(parts)}")

    # Compute class weights from TRAINING split only
    train_labels = df_train[label_col].values
    weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    ratio = weights[1] / weights[0]
    print(f"\n  Class weights (from train only): LGG={weights[0]:.3f}, HGG={weights[1]:.3f}")
    print(f"  Imbalance ratio: {ratio:.2f}x  {'⚠ Significant — use weighted loss' if ratio > 2 else '✓ Acceptable'}")
    print("="*55 + "\n")

    return {"LGG": weights[0], "HGG": weights[1]}


# ================================================================== #
# CHECK 2 — No data leakage                                          #
# ================================================================== #

def check_no_leakage(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
) -> None:
    """
    Assert zero subject overlap between splits.
    Hard fails with a clear message if leakage exists.
    """
    print("="*55)
    print("CHECK 2 — Data Leakage")
    print("="*55)

    train_ids = set(df_train["subject_id"])
    val_ids   = set(df_val["subject_id"])
    test_ids  = set(df_test["subject_id"])

    pairs = [
        ("Train ∩ Val",  train_ids & val_ids),
        ("Train ∩ Test", train_ids & test_ids),
        ("Val ∩ Test",   val_ids   & test_ids),
    ]

    leakage_found = False
    for label, overlap in pairs:
        if len(overlap) > 0:
            print(f"  ✗ LEAKAGE DETECTED — {label}: {len(overlap)} subjects")
            print(f"    Subjects: {list(overlap)[:5]} ...")
            leakage_found = True
        else:
            print(f"  ✓ {label}: 0 overlap")

    if leakage_found:
        raise AssertionError("Data leakage detected. Fix splits before proceeding.")

    print("  → All splits clean.")
    print("="*55 + "\n")


# ================================================================== #
# CHECK 3 — Overfit test                                             #
# ================================================================== #

def check_overfit_small_batch(
    model: nn.Module,
    dataset,
    n_samples:   int   = 5,
    n_epochs:    int   = 20,
    lr:          float = 1e-3,
    target_loss: float = 0.1,
    device:      str   = "cuda",
) -> bool:
    """
    Train model on n_samples for n_epochs.
    Loss should drop to near zero — if not, something is wrong
    with the data loader, label mapping, or model forward pass.

    Args:
        model       : your 3D model (before any real training)
        dataset     : BraTSDataset instance (train split)
        n_samples   : number of samples to overfit on (default 5)
        n_epochs    : training epochs (default 20)
        lr          : learning rate for this test (default 1e-3)
        target_loss : loss threshold to pass the check (default 0.1)
        device      : 'cuda' or 'cpu'

    Returns:
        True if overfit successful, False otherwise
    """
    print("="*55)
    print("CHECK 3 — Overfit Test (5 samples, 20 epochs)")
    print("="*55)

    model = model.to(device)
    subset    = Subset(dataset, list(range(n_samples)))
    loader    = DataLoader(subset, batch_size=n_samples, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = None

    for epoch in range(1, n_epochs + 1):
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} — loss: {final_loss:.4f}")

    passed = final_loss < target_loss
    print(f"\n  Final loss: {final_loss:.4f}  (target < {target_loss})")
    if passed:
        print("  ✓ PASSED — model can overfit. Data loader and model are correct.")
    else:
        print("  ✗ FAILED — model could not overfit 5 samples.")
        print("    Possible causes:")
        print("      - Label mapping is wrong (check GRADE_MAP in label_loader.py)")
        print("      - Data loader is returning all-zero volumes (check NIfTI paths)")
        print("      - Model forward pass has a bug (check output shape vs num_classes)")
        print("      - Learning rate too low for this test (try 1e-2)")
    print("="*55 + "\n")

    return passed


# ================================================================== #
# Run all checks                                                     #
# ================================================================== #

def run_all_checks(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    model:    nn.Module   = None,
    dataset               = None,
    device:   str         = "cuda",
) -> None:
    """
    Run checks 1 and 2 always.
    Run check 3 only if model and dataset are provided.
    """
    class_weights = check_label_distribution(df_train, df_val, df_test)
    check_no_leakage(df_train, df_val, df_test)

    if model is not None and dataset is not None:
        check_overfit_small_batch(model, dataset, device=device)
    else:
        print("[sanity_check] Skipping Check 3 — no model/dataset provided.")
        print("  Run check_overfit_small_batch() once your model is initialized.\n")

    return class_weights
