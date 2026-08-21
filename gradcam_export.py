"""
gradcam_export.py
-----------------
[LEGACY - BraTS 2021 Format Entry Point]

Generate and save Grad-CAM overlay images for representative test subjects.
NOTE: This script relies on src/ legacy loaders expecting BraTS 2021 CSV data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import DEVICE, GRADCAM_DIR, RESULTS_DIR

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("[gradcam_export] WARNING: opencv-python not installed. Overlays will be raw heatmap arrays.")

from src.train_models import create_dataloaders, get_model, get_model_path
from src.gradcam import generate_gradcam, get_target_layer, GradCAM3D
from src.label_loader import load_labels
from src.splits import make_splits
from src.brats_dataset import BraTSDataset
from src.data_transforms import get_val_transforms
from config import (BRATS_LABELS_CSV, BRATS_SUBJECTS_DIR, MODELS,
                    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)


def _load_checkpoint(model_name: str) -> torch.nn.Module:
    """Load a trained checkpoint from the model registry or default path."""
    import json
    from config import REGISTRY_PATH

    model = get_model(model_name).to(DEVICE)
    registry_path = REGISTRY_PATH
    ckpt_path = get_model_path(model_name)

    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        if model_name in registry:
            ckpt_path = Path(registry[model_name]["checkpoint_path"])

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for '{model_name}': {ckpt_path}\n"
            "Run training first on the GPU machine."
        )

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _pick_representative_subjects(df_test, n_subjects: int, label_col: str):
    """
    Pick balanced subjects from the test split.
    e.g. n_subjects=4 → 2 LGG, 2 HGG
    """
    per_class = max(1, n_subjects // 2)
    lgg = df_test[df_test[label_col] == 0].head(per_class)
    hgg = df_test[df_test[label_col] == 1].head(per_class)
    return df_test[df_test["subject_id"].isin(
        list(lgg["subject_id"]) + list(hgg["subject_id"])
    )].reset_index(drop=True)


def _save_overlay(overlay: np.ndarray, save_path: Path) -> None:
    """Save RGB uint8 numpy array as PNG."""
    if _CV2_AVAILABLE:
        import cv2
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    else:
        # Fallback: save as npy
        np.save(str(save_path).replace(".png", ".npy"), overlay)


def run_gradcam_export(n_subjects: int = 4, label_col: str = "label_grade") -> None:
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[gradcam_export] Loading test split …")
    df = load_labels(csv_path=BRATS_LABELS_CSV, data_root=BRATS_SUBJECTS_DIR)
    df_train, df_val, df_test = make_splits(
        df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO, stratify_col=label_col, random_seed=RANDOM_SEED,
    )
    df_subjects = _pick_representative_subjects(df_test, n_subjects, label_col)
    print(f"  Selected {len(df_subjects)} subjects: {df_subjects['subject_id'].tolist()}")

    val_transforms = get_val_transforms()

    for model_name in MODELS:
        print(f"\n[gradcam_export] Model: {model_name}")
        try:
            model = _load_checkpoint(model_name)
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping — {e}")
            continue

        for _, row in df_subjects.iterrows():
            sid   = row["subject_id"]
            label = int(row[label_col])
            label_name = "LGG" if label == 0 else "HGG"

            # Build a single-subject dataset to apply transforms
            single_df = df_subjects[df_subjects["subject_id"] == sid].copy()
            ds = BraTSDataset(single_df, transforms=val_transforms, label_col=label_col)
            sample = ds[0]
            inp = sample["image"].unsqueeze(0).to(DEVICE)   # (1, 4, D, H, W)

            try:
                overlay = generate_gradcam(model_name, model, inp)
                save_path = GRADCAM_DIR / f"{sid}_{label_name}_{model_name}.png"
                _save_overlay(overlay, save_path)
                print(f"  ✓ {sid} ({label_name}) → {save_path.name}")
            except Exception as exc:
                print(f"  ✗ {sid}: {exc}")

    print(f"\n[gradcam_export] All overlays saved to {GRADCAM_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Grad-CAM overlays for representative test subjects")
    parser.add_argument("--n_subjects", type=int, default=4,
                        help="Number of test subjects to visualize (split evenly across classes)")
    parser.add_argument("--label-col", type=str, default="label_grade",
                        choices=["label_grade", "label_idh"])
    args = parser.parse_args()
    run_gradcam_export(n_subjects=args.n_subjects, label_col=args.label_col)
