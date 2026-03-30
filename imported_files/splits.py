"""
splits.py
---------
Patient-level stratified train / val / test split.

Key guarantee: a subject_id appears in exactly ONE split.
No slice-level or augmentation-level leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    test_ratio: float  = 0.15,
    stratify_col: str  = "label_grade",
    random_seed: int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test at the PATIENT level.

    Args:
        df            : DataFrame from label_loader.load_labels()
        train_ratio   : fraction for training   (default 0.70)
        val_ratio     : fraction for validation (default 0.15)
        test_ratio    : fraction for test       (default 0.15)
        stratify_col  : column to stratify on   (default 'label_grade')
        random_seed   : for reproducibility     (default 42)

    Returns:
        df_train, df_val, df_test  — three non-overlapping DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"
    assert stratify_col in df.columns, \
        f"stratify_col '{stratify_col}' not found in DataFrame"

    # ------------------------------------------------------------------ #
    # Step 1: split off test set                                          #
    # ------------------------------------------------------------------ #
    df_trainval, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[stratify_col],
        random_state=random_seed,
    )

    # ------------------------------------------------------------------ #
    # Step 2: split trainval into train and val                           #
    # val_ratio relative to the full dataset, so adjust                  #
    # ------------------------------------------------------------------ #
    adjusted_val = val_ratio / (train_ratio + val_ratio)

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=adjusted_val,
        stratify=df_trainval[stratify_col],
        random_state=random_seed,
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    return df_train, df_val, df_test


def print_split_summary(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str = "label_grade",
) -> None:
    """Print per-split class distribution."""
    label_names = {0: "LGG", 1: "HGG"}

    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    total = len(df_train) + len(df_val) + len(df_test)

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = split[label_col].value_counts().sort_index()
        dist = "  ".join(
            f"{label_names.get(k, k)}={v}" for k, v in counts.items()
        )
        print(f"{name:6s}: {len(split):4d} subjects ({100*len(split)/total:.1f}%)  |  {dist}")

    print("="*50 + "\n")


def verify_no_leakage(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """
    Assert zero overlap between splits.
    Raises AssertionError immediately if leakage is detected.
    """
    train_ids = set(df_train["subject_id"])
    val_ids   = set(df_val["subject_id"])
    test_ids  = set(df_test["subject_id"])

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids   & test_ids

    assert len(tv) == 0, f"DATA LEAKAGE: {len(tv)} subjects in both Train and Val: {tv}"
    assert len(tt) == 0, f"DATA LEAKAGE: {len(tt)} subjects in both Train and Test: {tt}"
    assert len(vt) == 0, f"DATA LEAKAGE: {len(vt)} subjects in both Val and Test: {vt}"

    print("[splits] No leakage detected — all splits are clean.")


if __name__ == "__main__":
    # Smoke test
    from label_loader import load_labels, print_label_summary

    df = load_labels(
        csv_path="BraTS2021_Training_Data_Labels.csv",
        data_root="BraTS2021_Training_Data"
    )
    print_label_summary(df)

    df_train, df_val, df_test = make_splits(df)
    print_split_summary(df_train, df_val, df_test)
    verify_no_leakage(df_train, df_val, df_test)
