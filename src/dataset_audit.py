"""
dataset_audit.py
----------------
Standalone dataset audit for BraTS 2021.
Run this FIRST — before any model training.

Answers:
  1. Which subjects have usable labels?
  2. Which labels are present (grade, IDH)?
  3. How imbalanced are the classes?
  4. Are there missing or ambiguous cases?
  5. Are train/val/test splits patient-level and leakage-free?

Output:
  - Human-readable tables to stdout
  - results/audit_report.json  (machine-readable)

Usage:
    python src/dataset_audit.py
    python src/dataset_audit.py --csv path/to/labels.csv --data_root path/to/subjects
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRATS_LABELS_CSV, BRATS_SUBJECTS_DIR, RESULTS_DIR, RANDOM_SEED

try:
    from .label_loader import load_labels, GRADE_MAP, IDH_MAP
    from .splits import make_splits, verify_no_leakage, print_split_summary
except ImportError:
    from label_loader import load_labels, GRADE_MAP, IDH_MAP
    from splits import make_splits, verify_no_leakage, print_split_summary


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _bar(count: int, total: int, width: int = 30) -> str:
    filled = round(width * count / max(total, 1))
    return "█" * filled + "░" * (width - filled)


def _section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# ------------------------------------------------------------------ #
# Audit steps                                                        #
# ------------------------------------------------------------------ #

def audit_raw_csv(csv_path: Path) -> dict:
    """Inspect the raw CSV before any filtering."""
    _section("STEP 1 — Raw CSV inspection")
    df_raw = pd.read_csv(csv_path)
    df_raw.columns = df_raw.columns.str.strip()

    total_rows = len(df_raw)
    print(f"  Rows in CSV        : {total_rows}")
    print(f"  Columns            : {list(df_raw.columns)}")

    grade_col_present = "Grade" in df_raw.columns
    idh_col_present   = "IDH"   in df_raw.columns

    grade_counts = {}
    if grade_col_present:
        vc = df_raw["Grade"].value_counts(dropna=False)
        grade_counts = vc.to_dict()
        print(f"\n  Grade column values:")
        for val, cnt in vc.items():
            print(f"    {str(val):15s} : {cnt}")
    else:
        print("  ⚠ 'Grade' column NOT found in CSV.")

    idh_counts = {}
    if idh_col_present:
        vc = df_raw["IDH"].value_counts(dropna=False)
        idh_counts = vc.to_dict()
        print(f"\n  IDH column values:")
        for val, cnt in vc.items():
            print(f"    {str(val):15s} : {cnt}")
    else:
        print("  ⚠ 'IDH' column NOT found in CSV — IDH task will be skipped.")

    ambiguous_grade = int(df_raw["Grade"].isna().sum()) if grade_col_present else 0
    print(f"\n  Rows with missing/null Grade : {ambiguous_grade}")

    return {
        "total_rows": total_rows,
        "columns": list(df_raw.columns),
        "grade_col_present": grade_col_present,
        "idh_col_present": idh_col_present,
        "grade_value_counts": {str(k): int(v) for k, v in grade_counts.items()},
        "idh_value_counts":   {str(k): int(v) for k, v in idh_counts.items()},
        "ambiguous_grade_rows": ambiguous_grade,
    }


def audit_disk_presence(df: pd.DataFrame, data_root: Path) -> dict:
    """Check which subject directories actually exist on disk."""
    _section("STEP 2 — Subject directory presence on disk")

    all_expected = df["subject_id"].tolist()
    present    = [s for s in all_expected if (data_root / s).exists()]
    missing    = [s for s in all_expected if s not in present]

    print(f"  Expected subjects  : {len(all_expected)}")
    print(f"  Found on disk      : {len(present)}")
    print(f"  Missing on disk    : {len(missing)}")

    if missing:
        print(f"\n  First 10 missing:")
        for s in missing[:10]:
            print(f"    {s}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    else:
        print("  ✓ All subject directories found on disk.")

    # For present subjects, check modality file completeness
    SUFFIXES = ["_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz", "_flair.nii.gz"]
    incomplete = []
    for sid in present[:len(present)]:  # check all present
        subdir = data_root / sid
        for sfx in SUFFIXES:
            if not (subdir / f"{sid}{sfx}").exists():
                incomplete.append(sid)
                break

    print(f"\n  Subjects with all 4 modality files : {len(present) - len(incomplete)}")
    print(f"  Subjects missing ≥1 modality file  : {len(incomplete)}")
    if incomplete:
        print(f"  First 5 incomplete: {incomplete[:5]}")

    return {
        "expected": len(all_expected),
        "on_disk":  len(present),
        "missing_dirs": missing,
        "incomplete_modalities": incomplete,
        "usable_subjects": len(present) - len(incomplete),
    }


def audit_label_distribution(df: pd.DataFrame) -> dict:
    """Grade and IDH distribution on the usable label set."""
    _section("STEP 3 — Label distribution (usable subjects only)")

    total = len(df)

    # Grade
    grade_counts = df["label_grade"].value_counts().sort_index()
    reverse_grade = {v: k for k, v in GRADE_MAP.items()}
    print("\n  Grade (primary task):")
    for label, cnt in grade_counts.items():
        name = reverse_grade.get(label, str(label))
        pct  = 100 * cnt / total
        bar  = _bar(cnt, total)
        print(f"    {name} (label={label}) : {cnt:4d}  ({pct:.1f}%)  {bar}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=df["label_grade"].values)
    ratio = cw[1] / cw[0]
    flag  = "⚠ Use weighted loss" if ratio > 2 else "✓ Acceptable"
    print(f"\n  Class weights : LGG={cw[0]:.3f}, HGG={cw[1]:.3f}  → imbalance ratio {ratio:.2f}x  {flag}")

    # IDH
    idh_valid = df["label_idh"].dropna()
    idh_grade_counts = {}
    reverse_idh = {v: k for k, v in IDH_MAP.items()}
    print(f"\n  IDH (secondary task) — {len(idh_valid)}/{total} subjects have labels:")
    if len(idh_valid) > 0:
        idh_vc = idh_valid.value_counts().sort_index()
        for label, cnt in idh_vc.items():
            name = reverse_idh.get(int(label), str(label))
            pct  = 100 * cnt / len(idh_valid)
            bar  = _bar(int(cnt), len(idh_valid))
            print(f"    {name} (label={int(label)}) : {int(cnt):4d}  ({pct:.1f}%)  {bar}")
            idh_grade_counts[name] = int(cnt)
        idh_cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=idh_valid.values.astype(int))
        idh_ratio = idh_cw[1] / idh_cw[0]
        print(f"  IDH class weights : Mutant={idh_cw[0]:.3f}, Wildtype={idh_cw[1]:.3f}  → ratio {idh_ratio:.2f}x")
        idh_clean = len(idh_valid) == total
        print(f"  IDH label completeness : {'✓ All subjects have IDH' if idh_clean else f'⚠ {total - len(idh_valid)} subjects missing IDH'}")
    else:
        print("    No IDH labels found → IDH task CANNOT be used.")
        idh_clean = False

    return {
        "total_subjects": total,
        "grade": {
            "LGG": int(grade_counts.get(0, 0)),
            "HGG": int(grade_counts.get(1, 0)),
            "class_weights": cw.tolist(),
            "imbalance_ratio": float(ratio),
            "recommend_weighted_loss": bool(ratio > 2),
        },
        "idh": {
            "subjects_with_labels": int(len(idh_valid)),
            "subjects_missing": int(total - len(idh_valid)),
            "distribution": idh_grade_counts,
            "idh_task_feasible": bool(len(idh_valid) > 0),
            "idh_task_fully_labeled": bool(idh_clean),
        },
    }


def audit_splits(df: pd.DataFrame) -> dict:
    """Create splits and verify leakage-free, stratified."""
    _section("STEP 4 — Patient-level splits")

    df_train, df_val, df_test = make_splits(df, stratify_col="label_grade", random_seed=RANDOM_SEED)
    print_split_summary(df_train, df_val, df_test)

    # Leakage check
    try:
        verify_no_leakage(df_train, df_val, df_test)
        leakage = False
        print("  ✓ No leakage detected — all splits are patient-level and clean.")
    except AssertionError as e:
        leakage = True
        print(f"  ✗ LEAKAGE DETECTED: {e}")

    label_names = {0: "LGG", 1: "HGG"}
    def split_stats(split_df, name):
        vc = split_df["label_grade"].value_counts().sort_index()
        return {
            "n": len(split_df),
            "pct": round(100 * len(split_df) / len(df), 1),
            "LGG": int(vc.get(0, 0)),
            "HGG": int(vc.get(1, 0)),
        }

    return {
        "leakage_free": not leakage,
        "train": split_stats(df_train, "Train"),
        "val":   split_stats(df_val,   "Val"),
        "test":  split_stats(df_test,  "Test"),
    }


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #

def run_audit(csv_path: Path, data_root: Path) -> dict:
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  BraTS 2021 — DATASET AUDIT                                  ║")
    print("╚" + "═" * 63 + "╝")
    print(f"  CSV       : {csv_path}")
    print(f"  Data root : {data_root}")

    report = {}

    # Step 1: raw CSV
    report["raw_csv"] = audit_raw_csv(csv_path)

    # Step 2: load clean label df (applies grade filter + disk check)
    _section("Loading clean label DataFrame …")
    df = load_labels(csv_path=csv_path, data_root=data_root)
    print(f"  Subjects after filtering : {len(df)}")

    # Step 3: disk presence (re-check on raw id list before label_loader filters)
    report["disk"] = audit_disk_presence(df, data_root)

    # Step 4: label distribution
    report["labels"] = audit_label_distribution(df)

    # Step 5: splits
    report["splits"] = audit_splits(df)

    # Decision
    _section("AUDIT VERDICT")
    grade_ok = report["labels"]["grade"]["LGG"] > 0 and report["labels"]["grade"]["HGG"] > 0
    idh_ok   = report["labels"]["idh"]["idh_task_feasible"]
    leak_ok  = report["splits"]["leakage_free"]

    print(f"  Grade task (LGG vs HGG) : {'✓ PROCEED' if grade_ok and leak_ok else '✗ ABORT'}")
    print(f"  IDH task   (optional)   : {'✓ PROCEED' if idh_ok else '⚠ SKIP — insufficient labels'}")
    print(f"  Split leakage check     : {'✓ CLEAN' if leak_ok else '✗ LEAKAGE — fix before training'}")

    report["verdict"] = {
        "grade_task_ok": bool(grade_ok and leak_ok),
        "idh_task_ok":   bool(idh_ok),
        "splits_clean":  bool(leak_ok),
    }

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "audit_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Full report saved → {out_path}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2021 Dataset Audit")
    parser.add_argument("--csv",       type=str, default=str(BRATS_LABELS_CSV),
                        help="Path to BraTS2021_Training_Data_Labels.csv")
    parser.add_argument("--data_root", type=str, default=str(BRATS_SUBJECTS_DIR),
                        help="Root dir containing BraTS2021_XXXXX/ subject folders")
    args = parser.parse_args()

    run_audit(Path(args.csv), Path(args.data_root))
