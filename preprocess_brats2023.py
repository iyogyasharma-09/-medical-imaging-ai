"""
preprocess_brats2023.py
-----------------------
Preprocessing entry point for the BraTS 2023 dataset.

This script replaces the BraTS 2021 pipeline and handles:
  - BraTS 2023 GLI (Glioma)    — 1,251 training + 219 validation subjects
  - BraTS 2023 MEN (Meningioma) — 1,000 training subjects
  - BraTS 2023 PED (Pediatric)  — 99 training subjects

BraTS 2023 filename convention (DIFFERENT from BraTS 2021):
  BraTS-GLI-00000-000/
    BraTS-GLI-00000-000-t1n.nii.gz   ← T1 native  (was _t1.nii.gz)
    BraTS-GLI-00000-000-t1c.nii.gz   ← T1 contrast (was _t1ce.nii.gz)
    BraTS-GLI-00000-000-t2w.nii.gz   ← T2 weighted (was _t2.nii.gz)
    BraTS-GLI-00000-000-t2f.nii.gz   ← T2 FLAIR    (was _flair.nii.gz)
    BraTS-GLI-00000-000-seg.nii.gz   ← Segmentation

Usage:
    python preprocess_brats2023.py [--task GLI|MEN|PED|ALL] [--output-dir OUTPUT_DIR]

Output:
    processed_dataset/brats2023/
        manifest.json          — all subjects with paths
        train_manifest.json    — 70% split
        val_manifest.json      — 15% split
        test_manifest.json     — 15% split
        label_summary.txt      — per-task subject counts
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Project root so we can import config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# BraTS 2023 dataset roots (actual paths on disk)
# ---------------------------------------------------------------------------
DATASET_DIR = BASE_DIR / "dataset"

BRATS2023_ROOTS = {
    "GLI_TRAIN": DATASET_DIR / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
                              / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    "GLI_VAL":   DATASET_DIR / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
                              / "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
    "MEN_TRAIN": DATASET_DIR / "ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData"
                              / "BraTS-MEN-Train",
    # MEN validation set (141 subjects, no seg file)
    "MEN_VAL":   DATASET_DIR / "ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData"
                              / "ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData",
    # BraTS-MEN-TRAIN-FIX-V4: 47 subjects with corrected annotations
    # These REPLACE their counterparts in MEN_TRAIN (all 47 IDs overlap)
    "MEN_FIX":   DATASET_DIR / "BraTS-MEN-TRAIN-FIX-V4",
    "PED_TRAIN": DATASET_DIR / "ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData"
                              / "ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData",
}

# ---------------------------------------------------------------------------
# BraTS 2023 modality file suffixes — note the separator is "-" not "_"
# ---------------------------------------------------------------------------
MODALITY_SUFFIXES_2023 = {
    "t1n": "-t1n.nii.gz",   # T1 native
    "t1c": "-t1c.nii.gz",   # T1 contrast-enhanced
    "t2w": "-t2w.nii.gz",   # T2 weighted
    "t2f": "-t2f.nii.gz",   # T2 FLAIR
}
SEG_SUFFIX = "-seg.nii.gz"
CHANNEL_ORDER = ["t1n", "t1c", "t2w", "t2f"]

# ---------------------------------------------------------------------------
# Task → integer label mapping (for multi-task or single-task classification)
# ---------------------------------------------------------------------------
TASK_LABEL = {"GLI": 0, "MEN": 1, "PED": 2}


# ===========================================================================
# Core discovery functions
# ===========================================================================

def discover_subjects(root_dir: Path, task: str, split: str = "train") -> list[dict]:
    """
    Scan a BraTS 2023 root directory and return a list of subject dicts.

    Each dict contains:
        subject_id : str  e.g. 'BraTS-GLI-00000-000'
        data_path  : str  absolute path to subject folder
        task       : str  'GLI' | 'MEN' | 'PED'
        split      : str  'train' | 'val'
        complete   : bool  all 4 modalities + seg present
        missing    : list  names of any missing files
    """
    if not root_dir.exists():
        print(f"  [WARNING] Directory not found, skipping: {root_dir}")
        return []

    subjects = []
    prefix = f"BraTS-{task}-"

    for subj_dir in sorted(root_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        if not subj_dir.name.startswith(prefix):
            continue

        subject_id = subj_dir.name
        record = {
            "subject_id": subject_id,
            "data_path":  str(subj_dir),
            "task":       task,
            "split":      split,
            "complete":   True,
            "missing":    [],
        }

        # Check all 4 modalities
        for modality, suffix in MODALITY_SUFFIXES_2023.items():
            fpath = subj_dir / f"{subject_id}{suffix}"
            if not fpath.exists():
                record["missing"].append(f"{modality}{suffix}")
                record["complete"] = False

        # Check segmentation
        seg_path = subj_dir / f"{subject_id}{SEG_SUFFIX}"
        if not seg_path.exists():
            record["missing"].append(f"seg{SEG_SUFFIX}")
            # Validation data may not have seg — don't mark incomplete for val
            if split == "train":
                record["complete"] = False

        subjects.append(record)

    return subjects


def validate_and_filter(subjects: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split subject list into complete and incomplete.
    Returns (complete, incomplete).
    """
    complete   = [s for s in subjects if s["complete"]]
    incomplete = [s for s in subjects if not s["complete"]]
    return complete, incomplete


# ===========================================================================
# Label assignment helpers
# ===========================================================================

def assign_task_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign integer labels based on task (GLI=0, MEN=1, PED=2).
    This is the fallback when no grade CSV is available.
    """
    df["label"] = df["task"].map(TASK_LABEL)
    return df


def load_gli_grade_mapping(mapping_xlsx: Path, df_gli: pd.DataFrame) -> pd.DataFrame:
    """
    Load GLI grade labels (LGG=0, HGG=1) from BraTS2023_2017_GLI_Mapping.xlsx.

    The Excel has columns: BraTS2023, BraTS2021, Cohort Name, Site No, Local ID.
    There is NO direct 'Grade' column — grade is inferred from the Cohort Name:
      - 'TCGA-LGG'              -> LGG (label=0)
      - All others              -> HGG (label=1):
          GBM, CPTAC-GBM, IvyGAP, ACRIN-FMISO-Brain, priorToBraTS2017
      - 'Private Collection'    -> HGG (label=1, majority class for that cohort)

    Returns df_gli with added columns:
        label_grade : int   0=LGG, 1=HGG  (NaN if subject not found in mapping)
        brats2021_id: str   BraTS 2021 subject ID (NaN if new subject)
        cohort      : str   Source cohort name
    """
    if not mapping_xlsx.exists():
        print(f"  [INFO] GLI mapping file not found at {mapping_xlsx}. Skipping grade labels.")
        df_gli["label_grade"] = float("nan")
        df_gli["brats2021_id"] = float("nan")
        df_gli["cohort"] = float("nan")
        return df_gli

    try:
        xls = pd.read_excel(mapping_xlsx)
        xls.columns = xls.columns.str.strip()
        print(f"  [INFO] GLI mapping loaded: {len(xls)} rows")

        # Confirm expected columns are present
        if "BraTS2023" not in xls.columns or "Cohort Name (if publicly available)" not in xls.columns:
            print(f"  [WARNING] Unexpected Excel columns: {list(xls.columns)}")
            df_gli["label_grade"]  = float("nan")
            df_gli["brats2021_id"] = float("nan")
            df_gli["cohort"]       = float("nan")
            return df_gli

        cohort_col = "Cohort Name (if publicly available)"

        # Grade rule:
        #   TCGA-LGG        -> 0 (LGG)
        #   Everything else -> 1 (HGG: GBM, CPTAC, IvyGAP, Private Collection, etc.)
        LGG_COHORTS = {"TCGA-LGG"}
        xls["label_grade"] = xls[cohort_col].apply(
            lambda c: 0 if str(c).strip() in LGG_COHORTS else 1
        )

        # Build lookup maps keyed on BraTS2023 subject ID
        grade_map   = xls.set_index("BraTS2023")["label_grade"].to_dict()
        brats21_map = xls.set_index("BraTS2023")["BraTS2021"].to_dict() if "BraTS2021" in xls.columns else {}
        cohort_map  = xls.set_index("BraTS2023")[cohort_col].to_dict()

        df_gli["label_grade"]  = df_gli["subject_id"].map(grade_map)
        df_gli["brats2021_id"] = df_gli["subject_id"].map(brats21_map)
        df_gli["cohort"]       = df_gli["subject_id"].map(cohort_map)

        mapped  = df_gli["label_grade"].notna().sum()
        lgg_cnt = (df_gli["label_grade"] == 0).sum()
        hgg_cnt = (df_gli["label_grade"] == 1).sum()
        print(f"  [INFO] Grade labels mapped for {mapped}/{len(df_gli)} GLI subjects")
        print(f"    LGG (0) : {lgg_cnt}")
        print(f"    HGG (1) : {hgg_cnt}")
        if mapped < len(df_gli):
            print(f"    Unmapped (NaN): {len(df_gli) - mapped}")

    except Exception as e:
        print(f"  [WARNING] Failed to load GLI mapping Excel: {e}")
        df_gli["label_grade"]  = float("nan")
        df_gli["brats2021_id"] = float("nan")
        df_gli["cohort"]       = float("nan")

    return df_gli


# ===========================================================================
# Split helpers
# ===========================================================================

def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    test_ratio:  float = 0.15,
    stratify_col: str  = "label",
    random_seed: int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified patient-level 70/15/15 split.
    Falls back to non-stratified if there is only one class.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    stratify = df[stratify_col] if df[stratify_col].nunique() > 1 else None

    df_train, df_temp = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=stratify,
        random_state=random_seed,
    )

    relative_val = val_ratio / (val_ratio + test_ratio)
    stratify_temp = df_temp[stratify_col] if (stratify is not None and df_temp[stratify_col].nunique() > 1) else None

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - relative_val),
        stratify=stratify_temp,
        random_state=random_seed,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ===========================================================================
# Main orchestration
# ===========================================================================

def main(task_filter: str = "ALL", output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = BASE_DIR / "processed_dataset" / "brats2023"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("BraTS 2023 Preprocessing Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1 — Discover all subjects                                      #
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Discovering subjects...")
    all_subjects: list[dict] = []

    discovery_map = {
        "GLI": [("GLI_TRAIN", "train"), ("GLI_VAL", "val")],
        # MEN_FIX is handled separately below — it overrides 47 subjects in MEN_TRAIN
        "MEN": [("MEN_TRAIN", "train"), ("MEN_VAL", "val")],
        "PED": [("PED_TRAIN", "train")],
    }

    tasks_to_process = ["GLI", "MEN", "PED"] if task_filter == "ALL" else [task_filter.upper()]

    for task in tasks_to_process:
        for root_key, split in discovery_map.get(task, []):
            root = BRATS2023_ROOTS[root_key]
            print(f"  Scanning {task} {split}: {root.name}")
            subjects = discover_subjects(root, task, split)
            print(f"    Found {len(subjects)} subject folders")
            all_subjects.extend(subjects)

    # ------------------------------------------------------------------ #
    # Apply BraTS-MEN-TRAIN-FIX-V4 overrides                            #
    # 47 subjects in this folder have corrected annotations and REPLACE  #
    # their counterparts discovered from MEN_TRAIN above.               #
    # ------------------------------------------------------------------ #
    if "MEN" in tasks_to_process:
        fix_root = BRATS2023_ROOTS["MEN_FIX"]
        if fix_root.exists():
            fix_subjects = discover_subjects(fix_root, "MEN", "train")
            fix_ids = {s["subject_id"]: s for s in fix_subjects}
            # Remove original versions of the 47 subjects from all_subjects
            before_fix = len(all_subjects)
            all_subjects = [s for s in all_subjects if s["subject_id"] not in fix_ids]
            replaced = before_fix - len(all_subjects)
            # Add the fixed versions
            all_subjects.extend(fix_subjects.values() if isinstance(fix_subjects, dict) else fix_subjects)
            print(f"  Applied MEN FIX-V4: replaced {replaced} subjects with corrected annotations")
            print(f"    FIX-V4 path: {fix_root.name}")
        else:
            print(f"  [WARNING] MEN FIX-V4 folder not found: {fix_root}")

    # ------------------------------------------------------------------ #
    # Step 2 — Validate completeness                                      #
    # ------------------------------------------------------------------ #
    print("\n[Step 2] Validating file completeness...")
    complete, incomplete = validate_and_filter(all_subjects)
    print(f"  Complete subjects   : {len(complete)}")
    print(f"  Incomplete subjects : {len(incomplete)}")
    if incomplete:
        print("  Incomplete samples (first 10):")
        for s in incomplete[:10]:
            print(f"    {s['subject_id']} — missing: {s['missing']}")

    # ------------------------------------------------------------------ #
    # Step 3 — Build DataFrame                                            #
    # ------------------------------------------------------------------ #
    print("\n[Step 3] Building manifest DataFrame...")
    df = pd.DataFrame(complete)
    df = assign_task_labels(df)

    # Optionally try to load GLI grade labels from the mapping Excel
    if "GLI" in tasks_to_process:
        mapping_xlsx = DATASET_DIR / "BraTS2023_2017_GLI_Mapping (1).xlsx"
        df_gli = df[df["task"] == "GLI"].copy()
        df_gli = load_gli_grade_mapping(mapping_xlsx, df_gli)
        df = df.merge(
            df_gli[["subject_id", "label_grade"]],
            on="subject_id",
            how="left",
        )
    else:
        df["label_grade"] = float("nan")

    print(f"  Total subjects in manifest: {len(df)}")
    print("\n  Task distribution:")
    for task, count in df["task"].value_counts().items():
        print(f"    {task}: {count}")

    # ------------------------------------------------------------------ #
    # Step 4 — Train/val/test split (training subjects only)             #
    # ------------------------------------------------------------------ #
    print("\n[Step 4] Creating splits...")
    df_train_pool = df[df["split"] == "train"].copy()
    df_val_official = df[df["split"] == "val"].copy()

    df_train, df_val_new, df_test = make_splits(
        df_train_pool,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_col="label",
        random_seed=42,
    )

    # Merge official val set into our val (if it exists)
    if len(df_val_official) > 0:
        # Official val has no seg, use separately — do not include in splits
        print(f"  Official validation set: {len(df_val_official)} subjects "
              f"(no seg, kept separate)")

    print(f"  Train   : {len(df_train)} subjects")
    print(f"  Val     : {len(df_val_new)} subjects")
    print(f"  Test    : {len(df_test)} subjects")

    # ------------------------------------------------------------------ #
    # Step 5 — Save manifests                                            #
    # ------------------------------------------------------------------ #
    print("\n[Step 5] Saving manifests...")

    def df_to_records(df: pd.DataFrame) -> list[dict]:
        return df.to_dict(orient="records")

    manifest_path = output_dir / "manifest.json"
    train_path    = output_dir / "train_manifest.json"
    val_path      = output_dir / "val_manifest.json"
    test_path     = output_dir / "test_manifest.json"
    val_off_path  = output_dir / "val_official_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(df_to_records(df), f, indent=2, default=str)
    with open(train_path, "w") as f:
        json.dump(df_to_records(df_train), f, indent=2, default=str)
    with open(val_path, "w") as f:
        json.dump(df_to_records(df_val_new), f, indent=2, default=str)
    with open(test_path, "w") as f:
        json.dump(df_to_records(df_test), f, indent=2, default=str)
    if len(df_val_official) > 0:
        with open(val_off_path, "w") as f:
            json.dump(df_to_records(df_val_official), f, indent=2, default=str)

    # ------------------------------------------------------------------ #
    # Step 6 — Label summary                                             #
    # ------------------------------------------------------------------ #
    print("\n[Step 6] Label summary...")
    summary_lines = [
        "=" * 60,
        "BraTS 2023 Preprocessing Summary",
        "=" * 60,
        f"Total complete subjects : {len(df)}",
        f"Train                   : {len(df_train)}",
        f"Val (our split)         : {len(df_val_new)}",
        f"Test                    : {len(df_test)}",
        f"Official val (no seg)   : {len(df_val_official)}",
        "",
        "Task distribution (train pool):",
    ]
    for task, count in df_train_pool["task"].value_counts().items():
        summary_lines.append(f"  {task}: {count}")

    if df["label_grade"].notna().any():
        summary_lines.append("\nGLI Grade labels (0=LGG, 1=HGG):")
        gli_df = df[df["task"] == "GLI"]
        for label, count in gli_df["label_grade"].value_counts().sort_index().items():
            summary_lines.append(f"  label={int(label)}: {count}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = output_dir / "label_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\n[DONE] Manifests saved to: {output_dir}")
    print("  Next step: run train.py with --manifest-dir pointing to this folder.")
    print("=" * 60)


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BraTS 2023 preprocessing — builds subject manifests for training."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ALL",
        choices=["ALL", "GLI", "MEN", "PED"],
        help="Which BraTS 2023 task to preprocess (default: ALL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to save manifests (default: processed_dataset/brats2023/)",
    )
    args = parser.parse_args()
    main(
        task_filter=args.task,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
