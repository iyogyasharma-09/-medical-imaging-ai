"""
Patient-level stratified train/val/test splitting for BraTS 2021 subject folders.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    stratify_col: str = "label_grade",
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test ratios must sum to 1.0")
    if stratify_col not in df.columns:
        raise ValueError(f"stratify_col '{stratify_col}' not found")

    df_trainval, df_test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[stratify_col],
        random_state=random_seed,
    )
    adjusted_val = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=adjusted_val,
        stratify=df_trainval[stratify_col],
        random_state=random_seed,
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def print_split_summary(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, label_col: str = "label_grade") -> None:
    label_names = {0: "LGG", 1: "HGG"}
    total = len(df_train) + len(df_val) + len(df_test)
    print("\n" + "=" * 50)
    print("SPLIT SUMMARY")
    print("=" * 50)
    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        counts = split[label_col].value_counts().sort_index()
        dist = "  ".join(f"{label_names.get(k, k)}={v}" for k, v in counts.items())
        print(f"{name:6s}: {len(split):4d} subjects ({100 * len(split) / total:.1f}%)  |  {dist}")
    print("=" * 50 + "\n")


def verify_no_leakage(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    train_ids = set(df_train["subject_id"])
    val_ids = set(df_val["subject_id"])
    test_ids = set(df_test["subject_id"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("Data leakage detected between train/val/test splits")
