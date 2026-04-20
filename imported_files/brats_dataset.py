"""
brats_dataset.py
----------------
MONAI-compatible Dataset class for BraTS 2021 full 3D volume pipeline.

Input per subject : 4 NIfTI files → [T1, T1ce, T2, FLAIR] stacked as
                    a single 4-channel 3D numpy array  (C, D, H, W)
                    where C=4, native D×H×W ≈ 240×240×155 before resampling.
Label             : integer from label_grade column (0=LGG, 1=HGG)

Windows note:
    num_workers defaults to 0 because Python multiprocessing on Windows
    does not handle NIfTI file handles well. If running on Linux/Mac with
    enough RAM, you can set num_workers=4 for faster loading.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from monai.transforms import Compose
from typing import Optional, Dict, Any


# Standard BraTS modality filename suffixes
MODALITY_SUFFIXES = {
    "t1":   "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2":   "_t2.nii.gz",
    "flair":"_flair.nii.gz",
}

# Channel order matches config.py line 31: [T1, T1ce, T2, FLAIR]
CHANNEL_ORDER = ["t1", "t1ce", "t2", "flair"]


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2021 classification.

    Args:
        df         : DataFrame with columns [subject_id, label_grade, data_path]
        transforms : MONAI Compose transform pipeline (see transforms.py)
        label_col  : which label column to use (default: 'label_grade')
        return_meta: if True, also returns subject_id in the batch dict
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Compose] = None,
        label_col: str = "label_grade",
        return_meta: bool = False,
    ):
        self.df         = df.reset_index(drop=True)
        self.transforms = transforms
        self.label_col  = label_col
        self.return_meta = return_meta

        self._validate()

    def _validate(self):
        required = {"subject_id", self.label_col, "data_path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Warn on NaN labels
        nan_count = self.df[self.label_col].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"{nan_count} subjects have NaN in '{self.label_col}'. "
                f"Filter these out before constructing the Dataset."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _load_volume(self, subject_path: str, subject_id: str) -> np.ndarray:
        """
        Load 4 modality NIfTI files and stack into (4, D, H, W) numpy array.
        Uses SimpleITK for robust NIfTI reading.
        """
        import SimpleITK as sitk

        subject_path = Path(subject_path)
        channels = []

        for modality in CHANNEL_ORDER:
            suffix   = MODALITY_SUFFIXES[modality]
            filepath = subject_path / f"{subject_id}{suffix}"

            if not filepath.exists():
                raise FileNotFoundError(
                    f"Missing modality file: {filepath}\n"
                    f"Expected structure: {{data_root}}/{{subject_id}}/{{subject_id}}_{{modality}}.nii.gz"
                )

            img   = sitk.ReadImage(str(filepath))
            array = sitk.GetArrayFromImage(img)   # (D, H, W)  float32
            channels.append(array.astype(np.float32))

        volume = np.stack(channels, axis=0)   # (4, D, H, W)
        return volume

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row        = self.df.iloc[idx]
        subject_id = row["subject_id"]
        label      = int(row[self.label_col])
        data_path  = row["data_path"]

        volume = self._load_volume(data_path, subject_id)   # (4, D, H, W)

        sample = {
            "image": volume,                          # numpy (4, D, H, W)
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        # After transforms, image should be a torch.Tensor
        if isinstance(sample["image"], np.ndarray):
            sample["image"] = torch.from_numpy(sample["image"])

        if self.return_meta:
            sample["subject_id"] = subject_id

        return sample


def build_dataloaders(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    train_transforms,
    val_transforms,
    batch_size:  int  = 1,
    num_workers: int  = 0,
    label_col:   str  = "label_grade",
):
    """
    Convenience function to build all three DataLoaders for BraTS 2021 3D.

    Args:
        train_transforms : augmented transform pipeline (get_train_transforms())
        val_transforms   : deterministic pipeline (get_val_transforms())
        batch_size       : 1 is safe on most GPUs for 128^3 volumes;
                           use 2 only on ≥24 GB VRAM.
        num_workers      : 0 on Windows (multiprocessing + NIfTI = issues);
                           set to 4 on Linux for faster data loading.
    """
    from torch.utils.data import DataLoader

    train_ds = BraTSDataset(df_train, transforms=train_transforms, label_col=label_col)
    val_ds   = BraTSDataset(df_val,   transforms=val_transforms,   label_col=label_col)
    test_ds  = BraTSDataset(df_test,  transforms=val_transforms,   label_col=label_col)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=True,   # avoids single-sample batches breaking BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,          # always batch_size=1 for test to track per-subject results
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader
