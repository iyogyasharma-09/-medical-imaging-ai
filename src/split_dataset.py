"""
Patient-aware train/val/test splitting for BraTS 3D volumes.
"""

import json
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TEST_MANIFEST, TRAIN_MANIFEST, VAL_MANIFEST

try:
    from .train_models import build_split_manifests
except ImportError:
    from train_models import build_split_manifests


def summarize_manifest(path: Path) -> Dict:
    records = json.loads(path.read_text())
    class_counts = {"lgg": 0, "hgg": 0}
    for record in records:
        class_counts["lgg" if record["target"] == 0 else "hgg"] += 1
    return {
        "path": str(path),
        "volumes": len(records),
        "class_counts": class_counts,
    }


def main(force: bool = False) -> Dict:
    build_split_manifests(force=force)
    summary = {
        "train": summarize_manifest(TRAIN_MANIFEST),
        "val": summarize_manifest(VAL_MANIFEST),
        "test": summarize_manifest(TEST_MANIFEST),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BraTS 3D train/val/test manifests")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
