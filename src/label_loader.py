"""
Load BraTS 2021 labels from the official training labels CSV.
Primary task: LGG vs HGG.
Secondary task: IDH mutation when available.
"""

from pathlib import Path

import pandas as pd

from config import BRATS_LABELS_CSV, BRATS_SUBJECTS_DIR

GRADE_MAP = {"LGG": 0, "HGG": 1}
IDH_MAP = {"Mutant": 0, "Wildtype": 1}


def load_labels(csv_path: str | Path = BRATS_LABELS_CSV, data_root: str | Path = BRATS_SUBJECTS_DIR) -> pd.DataFrame:
    csv_path = Path(csv_path)
    data_root = Path(data_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = {"BraTS21ID", "Grade"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing expected columns: {missing_cols}. Found columns: {list(df.columns)}")

    df["subject_id"] = df["BraTS21ID"].apply(
        lambda x: f"BraTS2021_{int(x):05d}" if str(x).isdigit() else str(x).strip()
    )

    df = df[df["Grade"].isin(GRADE_MAP)].copy()
    df["label_grade"] = df["Grade"].map(GRADE_MAP).astype(int)

    if "IDH" in df.columns:
        df["label_idh"] = df["IDH"].map(IDH_MAP)
    else:
        df["label_idh"] = float("nan")

    df["data_path"] = df["subject_id"].apply(lambda sid: str((data_root / sid).resolve()))
    df = df[df["data_path"].apply(lambda p: Path(p).exists())].copy()

    return df[["subject_id", "label_grade", "label_idh", "data_path"]].reset_index(drop=True)


def print_label_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print("LABEL SUMMARY")
    print("=" * 50)
    print(f"Total subjects : {len(df)}")

    print("\nGrade distribution:")
    counts = df["label_grade"].value_counts().sort_index()
    reverse_grade = {v: k for k, v in GRADE_MAP.items()}
    for label, count in counts.items():
        print(f"  {reverse_grade[label]} (label={label}) : {count}  ({100 * count / len(df):.1f}%)")

    idh_valid = df["label_idh"].dropna()
    print(f"\nIDH labels available: {len(idh_valid)}/{len(df)}")
    if len(idh_valid) > 0:
        reverse_idh = {v: k for k, v in IDH_MAP.items()}
        for label, count in idh_valid.value_counts().sort_index().items():
            print(f"  {reverse_idh[int(label)]} (label={int(label)}) : {int(count)}")
    print("=" * 50 + "\n")
