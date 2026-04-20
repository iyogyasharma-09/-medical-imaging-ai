"""
BraTS 2021 NIfTI subject-folder Dataset and DataLoader helpers.

Input from _load_volume(): numpy array (4, D, H, W)
  4 channels = [T1, T1ce, T2, FLAIR], native resolution ~240×240×155.
  The transform pipeline handles resampling, cropping, and resizing.

Windows note: num_workers defaults to 0 — NIfTI file handles do not
  survive Python multiprocessing spawning on Windows. Set num_workers=4
  only on Linux/Mac.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose
from torch.utils.data import DataLoader, Dataset

MODALITY_SUFFIXES = {
    "t1": "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
}
CHANNEL_ORDER = ["t1", "t1ce", "t2", "flair"]


class BraTSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms: Optional[Compose] = None, label_col: str = "label_grade", return_meta: bool = False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.label_col = label_col
        self.return_meta = return_meta
        required = {"subject_id", self.label_col, "data_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_volume(self, subject_path: str, subject_id: str) -> np.ndarray:
        import SimpleITK as sitk

        subject_path = Path(subject_path)
        channels = []
        for modality in CHANNEL_ORDER:
            filepath = subject_path / f"{subject_id}{MODALITY_SUFFIXES[modality]}"
            if not filepath.exists():
                raise FileNotFoundError(f"Missing modality file: {filepath}")
            img = sitk.ReadImage(str(filepath))
            channels.append(sitk.GetArrayFromImage(img).astype(np.float32))
        return np.stack(channels, axis=0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        subject_id = row["subject_id"]
        volume = self._load_volume(row["data_path"], subject_id)
        sample: Dict[str, Any] = {
            "image": volume,
            "label": torch.tensor(int(row[self.label_col]), dtype=torch.long),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        if isinstance(sample["image"], np.ndarray):
            sample["image"] = torch.from_numpy(sample["image"])
        if self.return_meta:
            sample["subject_id"] = subject_id
        return sample


def build_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    train_transforms,
    val_transforms,
    batch_size: int = 1,
    num_workers: int = 0,
    label_col: str = "label_grade",
):
    """
    Build train / val / test DataLoaders for BraTS 2021 full 3D volumes.

    Args:
        batch_size  : 1 is safe for 128^3 volumes on ≥16 GB VRAM;
                      use 2 only with 24+ GB VRAM.
        num_workers : 0 on Windows; 4 on Linux for speed.
    """
    train_ds = BraTSDataset(df_train, transforms=train_transforms, label_col=label_col)
    val_ds = BraTSDataset(df_val, transforms=val_transforms, label_col=label_col)
    test_ds = BraTSDataset(df_test, transforms=val_transforms, label_col=label_col)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers),
            drop_last=True,   # avoid single-sample batches breaking BatchNorm
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers),
        ),
        DataLoader(
            test_ds,
            batch_size=1,       # always 1 for test to track per-subject results
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=bool(num_workers),
        ),
    )
