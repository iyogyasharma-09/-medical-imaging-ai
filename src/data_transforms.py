"""
data_transforms.py
------------------
MONAI transform pipelines for BraTS 2021 full 3D volume classification.

Pipeline design notes for BraTS 2021:
  - Input from _load_volume(): numpy array shape (4, D, H, W)
      4 channels = [T1, T1ce, T2, FLAIR] (channel_dim=0)
  - EnsureChannelFirstd must use channel_dim='no_channel' because
      the numpy array ALREADY has a leading channel axis (axis 0).
      Using channel_dim=0 (the MONAI default for "first axis IS channel")
      would make MONAI treat the existing channel axis correctly, but the
      safer / explicit approach is channel_dim='no_channel' to tell MONAI
      "channels are already in the first dim, do not insert a new one."
  - After EnsureChannelFirstd: MetaTensor shape (4, D, H, W)
  - After Spacingd 1mm isotropic: shape changes with the volume
  - After CropForegroundd: background stripped, brain-only bounding box
  - After Resized: fixed (4, 128, 128, 128) for batching
  - After NormalizeIntensityd (nonzero + channel_wise): z-score per modality
      on brain voxels only — standard BraTS preprocessing

Spatial size:
  128^3 is the default. Change INPUT_DEPTH/HEIGHT/WIDTH in config.py to 96
  if you hit GPU OOM on a smaller machine.
"""

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Resized,
    Spacingd,
    ToTensord,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPATIAL_SIZE


def get_train_transforms() -> Compose:
    """
    BraTS 2021 training pipeline:
      1.  EnsureChannelFirstd  — confirm (4, D, H, W) layout (no new dim added)
      2.  Orientationd         — reorient to RAS (standard neuroimaging convention)
      3.  Spacingd             — resample to 1 mm isotropic voxels
      4.  CropForegroundd      — remove empty background slices (non-zero brain mask)
      5.  Resized              — resize to SPATIAL_SIZE cube (128^3 default)
      6.  NormalizeIntensityd  — per-channel z-score on non-zero (brain) voxels only
      7.  RandFlipd x3        — left-right, anterior-posterior, superior-inferior flips
      8.  RandRotate90d        — random 90° rotation (valid for 3D brain symmetry)
      9.  RandZoomd            — random isotropic zoom 0.9–1.1× (BraTS scale jitter)
      10. RandShiftIntensityd  — global intensity shift ±10%
      11. RandScaleIntensityd  — global intensity scale ±10%
      12. RandGaussianNoised   — additive Gaussian noise (std=0.05)
      13. EnsureTyped          — ensure torch-compatible dtypes
      14. ToTensord            — convert to torch.Tensor
    """
    return Compose([
        # channel_dim=0: numpy array from _load_volume() is (4, D, H, W).
        # channel_dim=0 tells MONAI axis-0 IS the channel axis → keeps (4, D, H, W).
        # Do NOT use 'no_channel' — that inserts an extra dim giving (1, 4, D, H, W).
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),   # 1 mm isotropic — BraTS 2021 standard
            mode="bilinear",
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=lambda x: x > 0,   # crop based on non-zero brain voxels
            margin=0,
        ),
        Resized(
            keys=["image"],
            spatial_size=SPATIAL_SIZE,   # from config: 128^3 (or 96^3 if OOM)
            mode="trilinear",
        ),
        # --- Intensity normalization (BraTS standard) ---
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True,       # normalize only over non-zero (brain) voxels
            channel_wise=True,  # separate z-score per modality (T1/T1ce/T2/FLAIR)
        ),
        # --- Spatial augmentation ---
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),   # L-R
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),   # A-P
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),   # S-I
        RandRotate90d(keys=["image"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandZoomd(
            keys=["image"],
            prob=0.3,
            min_zoom=0.9,
            max_zoom=1.1,
            mode="trilinear",
            keep_size=True,     # output stays at SPATIAL_SIZE after zoom
        ),
        # --- Intensity augmentation ---
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        # --- Type conversion ---
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    """
    BraTS 2021 validation / test pipeline (deterministic — no augmentation).
    Identical geometry + normalization steps as training, all random ops removed.
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
