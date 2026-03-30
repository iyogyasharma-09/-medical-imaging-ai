"""
MONAI transform pipelines for BraTS 2021 classification.
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
    Resized,
    Spacingd,
    ToTensord,
)

from config import INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH

SPATIAL_SIZE = (INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH)


def get_train_transforms() -> Compose:
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    return Compose([
        EnsureChannelFirstd(keys=["image"], channel_dim=0),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0),
        Resized(keys=["image"], spatial_size=SPATIAL_SIZE, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"]),
    ])
