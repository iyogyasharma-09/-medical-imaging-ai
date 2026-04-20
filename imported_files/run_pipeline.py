"""
run_pipeline.py
---------------
Entry point. Run this to validate the full data pipeline
before touching any model training code.

Usage:
    python run_pipeline.py \
        --csv BraTS2021_Training_Data_Labels.csv \
        --data_root BraTS2021_Training_Data

Expected output (if everything is correct):
    LABEL SUMMARY       — subject counts, grade distribution
    SPLIT SUMMARY       — train/val/test counts per class
    CHECK 1             — distribution + class weights
    CHECK 2             — zero leakage confirmed
    CHECK 3             — run separately once model is ready
"""

import argparse
from dataset import (
    load_labels,
    print_label_summary,
    make_splits,
    print_split_summary,
    run_all_checks,
    build_dataloaders,
    get_train_transforms,
    get_val_transforms,
)


def main(csv_path: str, data_root: str):

    # ------------------------------------------------------------------ #
    # Step 1 — Load labels                                               #
    # ------------------------------------------------------------------ #
    print("\n[Step 1] Loading labels from CSV...")
    df = load_labels(csv_path=csv_path, data_root=data_root)
    print_label_summary(df)

    # ------------------------------------------------------------------ #
    # Step 2 — Split                                                     #
    # ------------------------------------------------------------------ #
    print("[Step 2] Creating patient-level splits...")
    df_train, df_val, df_test = make_splits(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        stratify_col="label_grade",
        random_seed=42,
    )
    print_split_summary(df_train, df_val, df_test)

    # ------------------------------------------------------------------ #
    # Step 3 — Sanity checks 1 and 2                                     #
    # ------------------------------------------------------------------ #
    print("[Step 3] Running sanity checks...")
    class_weights = run_all_checks(df_train, df_val, df_test)

    # ------------------------------------------------------------------ #
    # Step 4 — Build DataLoaders                                         #
    # ------------------------------------------------------------------ #
    print("[Step 4] Building DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
        batch_size=1,        # 1 is safe for 128^3 volumes on most GPUs
        num_workers=0,       # 0 on Windows; set 4 on Linux for speed
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # ------------------------------------------------------------------ #
    # Step 5 — Load one batch and verify tensor shape                    #
    # ------------------------------------------------------------------ #
    print("\n[Step 5] Loading one batch to verify tensor shape...")
    batch = next(iter(train_loader))
    images = batch["image"]
    labels = batch["label"]
    print(f"  Image tensor shape : {tuple(images.shape)}")   # expect (1, 4, 128, 128, 128)
    print(f"  Label tensor shape : {tuple(labels.shape)}")   # expect (1,)
    print(f"  Label values       : {labels.tolist()}")
    print(f"  Image dtype        : {images.dtype}")
    print(f"  Image min/max      : {images.min():.3f} / {images.max():.3f}")

    assert images.shape[1] == 4, \
        f"Expected 4 channels (T1/T1ce/T2/FLAIR), got {images.shape[1]}"
    assert images.ndim == 5, \
        f"Expected 5D tensor (B,C,D,H,W), got {images.ndim}D"

    print("\n✓ Pipeline is fully functional.")
    print("  Next step: initialize your models and run check_overfit_small_batch().")
    print(f"  Class weights for loss function: {class_weights}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2021 Dataset Pipeline")
    parser.add_argument(
        "--csv",
        type=str,
        default="BraTS2021_Training_Data_Labels.csv",
        help="Path to BraTS2021_Training_Data_Labels.csv"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="BraTS2021_Training_Data",
        help="Root directory containing BraTS2021_XXXXX/ subject folders"
    )
    args = parser.parse_args()
    main(args.csv, args.data_root)
