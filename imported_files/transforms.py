"""
transforms.py
-------------
MONAI transform pipelines for BraTS 2021 classification.

Two pipelines:
  get_train_transforms() — normalization + spatial augmentation
  get_val_transforms()   — normalization only (deterministic)

All transforms operate on dict-format samples with keys "image" and "label".
Input image shape: (4, D, H, W)  — 4-channel 3D numpy array
"""

from monai.transforms import (
    Compose,
    ToTensord,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    CropForegroundd,
    Resized,
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)


# ------------------------------------------------------------------ #
# Target spatial size — adjust based on GPU memory                   #
# 128x128x128 is safe for batch_size=2 on 24GB VRAM                  #
# Use 96x96x96 if OOM on smaller GPUs                               #
# ------------------------------------------------------------------ #
SPATIAL_SIZE = (128, 128, 128)


def get_train_transforms() -> Compose:
    """
    Training pipeline:
      1. Ensure channel-first format
      2. Reorient to standard RAS orientation
      3. Resample to 1mm isotropic voxel spacing
      4. Crop foreground (removes empty background)
      5. Resize to fixed spatial size
      6. Per-channel z-score normalization (non-zero voxels only)
      7. Spatial augmentation (flips, rotations)
      8. Intensity augmentation (shift, scale, noise)
      9. Convert to tensor
    """
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=lambda x: x > 0,   # crop based on non-zero voxels
        ),
        Resized(
            keys=["image"],
            spatial_size=SPATIAL_SIZE,
            mode="trilinear",
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,       # normalize only non-zero (brain) voxels
            channel_wise=True,  # separate z-score per modality channel
        ),
        # --- Spatial augmentation ---
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        # --- Intensity augmentation ---
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    """
    Validation / test pipeline:
      Same as training EXCEPT no augmentation steps.
      Deterministic — same output every run.
    """
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=lambda x: x > 0,
        ),
        Resized(
            keys=["image"],
            spatial_size=SPATIAL_SIZE,
            mode="trilinear",
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,
            channel_wise=True,
        ),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])
