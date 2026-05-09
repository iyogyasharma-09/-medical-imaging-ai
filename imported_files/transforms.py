"""
transforms.py
-------------
MONAI transform pipelines for BraTS 2021 full 3D volume classification.

Two pipelines:
  get_train_transforms() — geometry normalization + augmentation
  get_val_transforms()   — geometry normalization only (deterministic)

All transforms operate on dict-format samples with keys "image" and "label".
Input image shape: (4, D, H, W)  — 4-channel 3D numpy array stacked as
                                    [T1, T1ce, T2, FLAIR] (channel axis = 0).

IMPORTANT — EnsureChannelFirstd note:
  _load_volume() returns a numpy array of shape (4, D, H, W) where channel=0.
  We use channel_dim='no_channel' to tell MONAI that channels are ALREADY in
  the leading dimension and it should NOT insert an extra dimension.
  Using the wrong channel_dim would produce a (1, 4, D, H, W) tensor (broken).

BraTS 2021 native voxel space: 240×240×155 @ 1 mm isotropic.
After pipeline: (4, SPATIAL_SIZE[0], SPATIAL_SIZE[1], SPATIAL_SIZE[2]).
"""


def _select_nonzero(x):
    """Brain mask selector — picklable named function (replaces lambda x: x > 0).
    Required for multiprocessing DataLoader workers on Windows (spawn context).
    """
    return x > 0


from monai.transforms import (
    Compose,
    ToTensord,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandZoomd,
    CropForegroundd,
    Resized,
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
)


# ------------------------------------------------------------------ #
# Target spatial size — read from config for single source of truth. #
# Fallback to 128^3 if run standalone (e.g. imported_files context). #
# ------------------------------------------------------------------ #
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import SPATIAL_SIZE
except ImportError:
    # Standalone fallback
    # 128^3 is safe for batch_size=1 on ≥16 GB VRAM
    # Use 96^3 if OOM on smaller GPUs
    SPATIAL_SIZE = (128, 128, 128)


def get_train_transforms() -> Compose:
    """
    BraTS 2021 training pipeline:
      1.  EnsureChannelFirstd  — confirm (4, D, H, W) — no new dim added
      2.  Orientationd         — reorient to RAS
      3.  Spacingd             — resample to 1 mm isotropic
      4.  CropForegroundd      — crop to brain bounding box (non-zero voxels)
      5.  Resized              — resize to SPATIAL_SIZE
      6.  NormalizeIntensityd  — z-score per modality on non-zero voxels
      7.  RandFlipd x3        — axial / coronal / sagittal flips
      8.  RandRotate90d        — random 90° in-plane rotation
      9.  RandZoomd            — random isotropic zoom 0.9–1.1×
      10. RandShiftIntensityd  — intensity offset ±10%
      11. RandScaleIntensityd  — intensity scale ±10%
      12. RandGaussianNoised   — additive noise (std=0.05, prob=20%)
      13. EnsureTyped + ToTensord
    """
    return Compose([
        # channel_dim='no_channel': numpy is already (4, D, H, W)
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),   # BraTS 2021: 1 mm isotropic
            mode="bilinear",
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=_select_nonzero,  # brain mask = non-zero voxels (picklable)
            margin=0,
        ),
        Resized(
            keys=["image"],
            spatial_size=SPATIAL_SIZE,
            mode="trilinear",
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,       # z-score on brain voxels only
            channel_wise=True,  # separate per modality (T1/T1ce/T2/FLAIR)
        ),
        # --- Spatial augmentation ---
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandZoomd(
            keys=["image"],
            prob=0.3,
            min_zoom=0.9,
            max_zoom=1.1,
            mode="trilinear",
            keep_size=True,
        ),
        # --- Intensity augmentation ---
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    """
    Validation / test pipeline — deterministic, no augmentation.
    Identical geometry normalization as training.
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
            select_fn=_select_nonzero,
            margin=0,
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
