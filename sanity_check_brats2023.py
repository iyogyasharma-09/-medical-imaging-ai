"""
sanity_check_brats2023.py
--------------------------
End-to-end pipeline sanity check for BraTS 2023.

Loads one NIfTI volume from the manifest through the full MONAI
transform pipeline and verifies output shape and dtype.

Run from c:\\mri\\mri-:
    python sanity_check_brats2023.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

MANIFEST = BASE_DIR / "processed_dataset" / "brats2023" / "train_manifest.json"
MODALITY_SUFFIXES = {
    "t1n": "-t1n.nii.gz",
    "t1c": "-t1c.nii.gz",
    "t2w": "-t2w.nii.gz",
    "t2f": "-t2f.nii.gz",
}
CHANNEL_ORDER = ["t1n", "t1c", "t2w", "t2f"]


def main():
    print("\n" + "=" * 55)
    print("BraTS 2023 End-to-End Sanity Check")
    print("=" * 55)

    # ------------------------------------------------------------------ #
    # Step 1 — Load manifest and pick a subject with grade label         #
    # ------------------------------------------------------------------ #
    if not MANIFEST.exists():
        print("[ERROR] Manifest not found. Run preprocess_brats2023.py first.")
        sys.exit(1)

    with open(MANIFEST) as f:
        records = json.load(f)

    # Prefer a subject with a grade label
    sample = next(
        (r for r in records
         if r.get("label_grade") is not None and str(r.get("label_grade")) != "nan"),
        records[0]
    )

    print(f"\nSubject    : {sample['subject_id']}")
    print(f"Task       : {sample['task']}")
    print(f"label      : {sample['label']}")
    print(f"label_grade: {sample.get('label_grade', 'N/A')}")
    print(f"cohort     : {sample.get('cohort', 'N/A')}")
    print(f"data_path  : {sample['data_path']}")

    # ------------------------------------------------------------------ #
    # Step 2 — List files in subject folder                              #
    # ------------------------------------------------------------------ #
    data_path = Path(sample["data_path"])
    print(f"\nFiles in subject folder:")
    for fname in sorted(os.listdir(data_path)):
        size_mb = os.path.getsize(data_path / fname) / (1024 * 1024)
        print(f"  {fname}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------ #
    # Step 3 — Load NIfTI volumes with SimpleITK                        #
    # ------------------------------------------------------------------ #
    import SimpleITK as sitk
    print("\nLoading modality volumes...")

    subject_id = sample["subject_id"]
    channels = []
    for mod in CHANNEL_ORDER:
        fpath = data_path / f"{subject_id}{MODALITY_SUFFIXES[mod]}"
        img   = sitk.ReadImage(str(fpath))
        arr   = sitk.GetArrayFromImage(img).astype("float32")
        channels.append(arr)
        nonzero = np.count_nonzero(arr)
        print(f"  {mod}: shape={arr.shape}, min={arr.min():.1f}, "
              f"max={arr.max():.1f}, nonzero={nonzero:,}")

    volume = np.stack(channels, axis=0)
    print(f"\nStacked volume: {volume.shape}  (expected: (4, ~155, ~240, ~240))")
    assert volume.shape[0] == 4, f"Expected 4 channels, got {volume.shape[0]}"

    # ------------------------------------------------------------------ #
    # Step 4 — Run through MONAI val transform pipeline                 #
    # ------------------------------------------------------------------ #
    print("\nApplying val transform pipeline...")
    from imported_files.transforms import get_val_transforms, SPATIAL_SIZE
    transforms = get_val_transforms()

    sample_dict = {
        "image": volume,
        "label": torch.tensor(int(sample["label"]), dtype=torch.long),
    }
    out = transforms(sample_dict)
    img_tensor = out["image"]

    print(f"\nAfter transforms:")
    print(f"  image shape : {tuple(img_tensor.shape)}")
    print(f"  expected    : (4, {SPATIAL_SIZE[0]}, {SPATIAL_SIZE[1]}, {SPATIAL_SIZE[2]})")
    print(f"  image dtype : {img_tensor.dtype}")
    print(f"  image min   : {img_tensor.min():.4f}")
    print(f"  image max   : {img_tensor.max():.4f}")
    print(f"  image mean  : {img_tensor.mean():.4f}")
    print(f"  label value : {out['label']}")

    # ------------------------------------------------------------------ #
    # Step 5 — Assertions                                                #
    # ------------------------------------------------------------------ #
    assert img_tensor.ndim == 4, f"Expected 4D tensor (C,D,H,W), got {img_tensor.ndim}D"
    assert img_tensor.shape[0] == 4, f"Expected 4 channels, got {img_tensor.shape[0]}"
    assert img_tensor.shape[1:] == SPATIAL_SIZE, \
        f"Expected spatial size {SPATIAL_SIZE}, got {tuple(img_tensor.shape[1:])}"
    assert img_tensor.dtype == torch.float32, f"Expected float32, got {img_tensor.dtype}"

    print("\n[PASS] All assertions passed.")
    print("[PASS] BraTS 2023 pipeline is fully functional.")
    print("=" * 55)


if __name__ == "__main__":
    main()
