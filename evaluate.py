"""
evaluate.py
-----------
[LEGACY - BraTS 2021 Evaluation Entry Point]

NOTE: For active BraTS 2023 research, use:
    python evaluate_models.py

This script delegates to src/evaluate_models.py for BraTS 2021 legacy evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import create_directories
from src.evaluate_models import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate 3D BraTS 2021 ensemble (single / simple-avg / reliability-weighted)"
    )
    parser.add_argument(
        "--label-col", type=str, default="label_grade",
        choices=["label_grade", "label_idh"],
        help="Which label column to evaluate on"
    )
    args = parser.parse_args()

    create_directories()

    print("=" * 60)
    print("  BraTS 2021 — Ensemble Evaluation")
    print("=" * 60)
    print(f"  Label col : {args.label_col}\n")

    summary = evaluate(label_col=args.label_col)
    print("\n  Full results:")
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "single_models"},
        indent=2
    ))


if __name__ == "__main__":
    main()
