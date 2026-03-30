"""
label_loader.py
---------------
Loads BraTS 2021 labels from the official CSV.
Primary task: LGG vs HGG (Grade)
Secondary task: IDH mutation (loaded but not used until Phase 3)
"""

import pandas as pd
from pathlib import Path


GRADE_MAP = {"LGG": 0, "HGG": 1}
IDH_MAP   = {"Mutant": 0, "Wildtype": 1}


def load_labels(csv_path: str, data_root: str) -> pd.DataFrame:
    """
    Parse BraTS2021_Training_Data_Labels.csv and return a clean DataFrame.

    Columns returned:
        subject_id   : str  e.g. 'BraTS2021_00000'
        label_grade  : int  0=LGG, 1=HGG   (primary task)
        label_idh    : int  0=Mutant, 1=Wildtype  (NaN if missing)
        data_path    : str  absolute path to subject folder

    Args:
        csv_path  : path to BraTS2021_Training_Data_Labels.csv
        data_root : root directory containing BraTS2021_XXXXX/ subject folders
    """
    csv_path  = Path(csv_path)
    data_root = Path(data_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------ #
    # Normalise column names — BraTS CSVs sometimes have leading spaces   #
    # ------------------------------------------------------------------ #
    df.columns = df.columns.str.strip()

    # Expected columns: BraTS21ID, Grade, IDH  (verify early)
    required = {"BraTS21ID", "Grade"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing expected columns: {missing_cols}\n"
                         f"Found columns: {list(df.columns)}")

    # ------------------------------------------------------------------ #
    # Subject ID → zero-padded string e.g. 'BraTS2021_00000'             #
    # ------------------------------------------------------------------ #
    df["subject_id"] = df["BraTS21ID"].apply(
        lambda x: f"BraTS2021_{int(x):05d}" if str(x).isdigit() else str(x).strip()
    )

    # ------------------------------------------------------------------ #
    # Grade label (primary) — drop rows with missing/unknown grade        #
    # ------------------------------------------------------------------ #
    valid_grades = set(GRADE_MAP.keys())
    before = len(df)
    df = df[df["Grade"].isin(valid_grades)].copy()
    after = len(df)
    if before != after:
        print(f"[label_loader] Dropped {before - after} rows with missing/invalid Grade.")

    df["label_grade"] = df["Grade"].map(GRADE_MAP).astype(int)

    # ------------------------------------------------------------------ #
    # IDH label (secondary) — keep NaN for missing, don't drop rows       #
    # ------------------------------------------------------------------ #
    if "IDH" in df.columns:
        df["label_idh"] = df["IDH"].map(IDH_MAP)   # NaN stays NaN
    else:
        print("[label_loader] WARNING: 'IDH' column not found. label_idh set to NaN.")
        df["label_idh"] = float("nan")

    # ------------------------------------------------------------------ #
    # Attach data paths and check existence                               #
    # ------------------------------------------------------------------ #
    df["data_path"] = df["subject_id"].apply(lambda sid: str(data_root / sid))

    missing_dirs = df[~df["data_path"].apply(lambda p: Path(p).exists())]
    if len(missing_dirs) > 0:
        print(f"[label_loader] WARNING: {len(missing_dirs)} subject folders not found on disk.")
        print(f"  First few missing: {missing_dirs['subject_id'].head(5).tolist()}")
        df = df[df["data_path"].apply(lambda p: Path(p).exists())].copy()
        print(f"[label_loader] Kept {len(df)} subjects with existing folders.")

    df = df[["subject_id", "label_grade", "label_idh", "data_path"]].reset_index(drop=True)

    return df


def print_label_summary(df: pd.DataFrame) -> None:
    """Print a quick sanity summary of the loaded labels."""
    print("\n" + "="*50)
    print("LABEL SUMMARY")
    print("="*50)
    print(f"Total subjects : {len(df)}")
    print("\nGrade distribution (primary task):")
    counts = df["label_grade"].value_counts().sort_index()
    for k, v in counts.items():
        name = {v: k for k, v in GRADE_MAP.items()}[k]
        print(f"  {name} (label={k}) : {v}  ({100*v/len(df):.1f}%)")

    idh_valid = df["label_idh"].dropna()
    print(f"\nIDH distribution (secondary task, {len(idh_valid)}/{len(df)} subjects have labels):")
    if len(idh_valid) > 0:
        counts_idh = idh_valid.value_counts().sort_index()
        for k, v in counts_idh.items():
            name = {v: k for k, v in IDH_MAP.items()}[int(k)]
            print(f"  {name} (label={int(k)}) : {int(v)}  ({100*v/len(idh_valid):.1f}%)")
    print("="*50 + "\n")


if __name__ == "__main__":
    # Quick smoke test — update paths to your environment
    df = load_labels(
        csv_path="BraTS2021_Training_Data_Labels.csv",
        data_root="BraTS2021_Training_Data"
    )
    print_label_summary(df)
