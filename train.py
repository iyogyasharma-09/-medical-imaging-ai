"""
train.py
--------
[LEGACY - BraTS 2021 Training Entry Point]

NOTE: For active BraTS 2023 research, use:
    python train_brats2023.py

This script calls src/train_models.py::train_all_models() directly for BraTS 2021 legacy data.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import EPOCHS, BATCH_SIZE, MODELS, create_directories
from src.train_models import train_all_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D BraTS 2021 LGG-vs-HGG reliability-aware ensemble"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=list(MODELS.keys()),
        help="Which models to train (default: all 3)"
    )
    parser.add_argument("--epochs",     type=int, default=EPOCHS,     help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size (1–2 recommended for 3D)")
    parser.add_argument(
        "--label-col", type=str, default="label_grade",
        choices=["label_grade", "label_idh"],
        help="Which label to train on"
    )
    args = parser.parse_args()

    create_directories()

    print("=" * 60)
    print("  BraTS 2021 — 3D Reliability-Aware Ensemble Training")
    print("=" * 60)
    print(f"  Models     : {args.models or list(MODELS.keys())}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Label col  : {args.label_col}")
    print("=" * 60 + "\n")

    summary = train_all_models(
        model_names=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_col=args.label_col,
    )

    print("\n" + "=" * 60)
    print("  Training complete. Summary:")
    print("=" * 60)
    for name, result in summary.items():
        print(f"  {name:<30} best_val_acc={result['best_val_accuracy']:.4f}  "
              f"uncertainty={result['mean_uncertainty']:.4f}  "
              f"epochs={result['epochs_trained']}")
    print("\n  Checkpoints → models/")
    print("  Registry    → results/model_registry.json")
    print("  History     → results/<model>_history.json\n")


if __name__ == "__main__":
    main()
