"""
Pipeline smoke test for the BraTS 2021 NIfTI-based data path.
"""

import argparse

from brats_dataset import build_dataloaders
from data_transforms import get_train_transforms, get_val_transforms
from label_loader import load_labels, print_label_summary
from sanity_check import run_all_checks
from splits import make_splits, print_split_summary


def main() -> None:
    df = load_labels()
    print_label_summary(df)
    df_train, df_val, df_test = make_splits(df)
    print_split_summary(df_train, df_val, df_test)
    class_weights = run_all_checks(df_train, df_val, df_test)
    train_loader, val_loader, test_loader = build_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
        batch_size=2,
        num_workers=0,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    batch = next(iter(train_loader))
    print(f"Image tensor shape: {tuple(batch['image'].shape)}")
    print(f"Label tensor shape: {tuple(batch['label'].shape)}")
    print(f"Class weights: {class_weights}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2021 pipeline smoke test")
    parser.parse_args()
    main()
