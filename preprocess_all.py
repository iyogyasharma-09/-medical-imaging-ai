"""
Compatibility entry point for preparing the BraTS 3D dataset.

The project no longer merges legacy 2D image archives into `processed_dataset/`.
Instead, it reconstructs patient volumes from raw BraTS H5 slices and writes
patient-aware train/val/test manifests.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import BRATS_METADATA_CSV, BRATS_RAW_DIR
from src.split_dataset import main as build_split_summary


def validate_inputs() -> None:
    missing = []
    if not BRATS_METADATA_CSV.exists():
        missing.append(str(BRATS_METADATA_CSV))
    if not BRATS_RAW_DIR.exists():
        missing.append(str(BRATS_RAW_DIR))
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"BraTS input files are missing:\n{joined}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BraTS 3D manifests for the LGG-vs-HGG pipeline")
    parser.add_argument("--force", action="store_true", help="Rebuild manifests even if they already exist")
    args = parser.parse_args()

    validate_inputs()
    build_split_summary(force=args.force)
