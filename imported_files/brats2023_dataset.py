"""
brats_dataset.py  (BraTS 2023 edition)
----------------------------------------
MONAI-compatible Dataset for BraTS 2023 full 3D volume pipeline.

BraTS 2023 filename convention:
  Subject folder : BraTS-GLI-00000-000/
  T1 native      : BraTS-GLI-00000-000-t1n.nii.gz   (ch 0)
  T1 contrast    : BraTS-GLI-00000-000-t1c.nii.gz   (ch 1)
  T2 weighted    : BraTS-GLI-00000-000-t2w.nii.gz   (ch 2)
  T2 FLAIR       : BraTS-GLI-00000-000-t2f.nii.gz   (ch 3)

Input per subject : 4 NIfTI files → stacked as (C=4, D, H, W) numpy array
Label             : integer from 'label' column in the manifest DataFrame

Windows note:
    num_workers defaults to 0 because Python multiprocessing on Windows
    does not handle NIfTI file handles well. On Linux/Mac set num_workers=4.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# BraTS 2023 modality suffixes — separator is "-" (NOT "_" as in BraTS 2021)
# ---------------------------------------------------------------------------
MODALITY_SUFFIXES = {
    "t1n": "-t1n.nii.gz",   # T1 native       (was _t1.nii.gz   in BraTS 2021)
    "t1c": "-t1c.nii.gz",   # T1 contrast-enh (was _t1ce.nii.gz in BraTS 2021)
    "t2w": "-t2w.nii.gz",   # T2 weighted     (was _t2.nii.gz   in BraTS 2021)
    "t2f": "-t2f.nii.gz",   # T2 FLAIR        (was _flair.nii.gz in BraTS 2021)
}

# Channel order: [T1n, T1c, T2w, T2f]
CHANNEL_ORDER = ["t1n", "t1c", "t2w", "t2f"]


# ===========================================================================
# Dataset class
# ===========================================================================

class BraTS2023Dataset(Dataset):
    """
    PyTorch Dataset for BraTS 2023 classification.

    Args:
        df          : DataFrame with columns [subject_id, label, data_path]
                      (produced by preprocess_brats2023.py)
        transforms  : MONAI Compose transform pipeline (see transforms.py)
        label_col   : column name for the integer label (default: 'label')
        return_meta : if True, also return subject_id in the batch dict
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Compose] = None,
        label_col: str = "label",
        return_meta: bool = False,
    ):
        self.df          = df.reset_index(drop=True)
        self.transforms  = transforms
        self.label_col   = label_col
        self.return_meta = return_meta
        self._validate()

    def _validate(self):
        required = {"subject_id", self.label_col, "data_path"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

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
                    f"Expected: {{data_path}}/{{subject_id}}{{suffix}}\n"
                    f"Available suffixes: {list(MODALITY_SUFFIXES.values())}"
                )

            img   = sitk.ReadImage(str(filepath))
            array = sitk.GetArrayFromImage(img)        # (D, H, W) float32
            channels.append(array.astype(np.float32))

        volume = np.stack(channels, axis=0)            # (4, D, H, W)
        return volume

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row        = self.df.iloc[idx]
        subject_id = row["subject_id"]
        label      = int(row[self.label_col])
        data_path  = row["data_path"]

        volume = self._load_volume(data_path, subject_id)  # (4, D, H, W)

        sample: Dict[str, Any] = {
            "image": volume,                               # numpy (4, D, H, W)
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        if isinstance(sample["image"], np.ndarray):
            sample["image"] = torch.from_numpy(sample["image"])

        if self.return_meta:
            sample["subject_id"] = subject_id

        return sample


# ===========================================================================
# Manifest loading helpers
# ===========================================================================

def load_manifest(manifest_path: str | Path, label_col: str = "label") -> pd.DataFrame:
    """
    Load a JSON manifest produced by preprocess_brats2023.py into a DataFrame.
    Drops rows with NaN labels automatically.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    before = len(df)
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"[load_manifest] Dropped {before - after} rows with NaN '{label_col}'.")
    return df


# ===========================================================================
# DataLoader builder
# ===========================================================================

def build_dataloaders(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    train_transforms,
    val_transforms,
    batch_size:  int = 1,
    num_workers: int = 0,
    label_col:   str = "label",
) -> tuple:
    """
    Build train / val / test DataLoaders for BraTS 2023 full 3D volumes.

    Args:
        batch_size  : 1 is safe for 96³ volumes on 8 GB VRAM;
                      use 2 only on ≥16 GB VRAM.
        num_workers : 0 on Windows (NIfTI + multiprocessing = issues);
                      set to 4 on Linux for faster loading.
    """
    train_ds = BraTS2023Dataset(df_train, transforms=train_transforms, label_col=label_col)
    val_ds   = BraTS2023Dataset(df_val,   transforms=val_transforms,   label_col=label_col)
    test_ds  = BraTS2023Dataset(df_test,  transforms=val_transforms,   label_col=label_col)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
        drop_last=True,   # avoid single-sample batches breaking BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,       # always 1 for test to track per-subject results
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers),
    )

    return train_loader, val_loader, test_loader


# ===========================================================================
# Quick smoke test
# ===========================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    manifest_dir = Path(__file__).parent / "processed_dataset" / "brats2023"
    train_path   = manifest_dir / "train_manifest.json"

    if not train_path.exists():
        print("[ERROR] Run preprocess_brats2023.py first to generate manifests.")
        sys.exit(1)

    df = load_manifest(train_path, label_col="label")
    print(f"Loaded {len(df)} training subjects")
    print(df.head())
    print("Label distribution:", df["label"].value_counts().to_dict())
