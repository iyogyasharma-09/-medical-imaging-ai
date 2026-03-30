"""
Single-volume inference for the true 3D reliability-aware ensemble.

Migration notes:
- Removed the old volume_id / H5 slice-stacking inference path.
- Added subject_id input for BraTS 2021 subject folders.
- Uses the BraTS 2021 NIfTI loader plus the same val/test transforms as training.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BRATS_SUBJECTS_DIR, DEVICE, GRADCAM_DIR

try:
    from .data_transforms import get_val_transforms
    from .uncertainty import ReliabilityWeightedEnsemble
except ImportError:
    from data_transforms import get_val_transforms
    from uncertainty import ReliabilityWeightedEnsemble

_MODALITY_SUFFIXES = {
    "t1": "_t1.nii.gz",
    "t1ce": "_t1ce.nii.gz",
    "t2": "_t2.nii.gz",
    "flair": "_flair.nii.gz",
}
_CHANNEL_ORDER = ["t1", "t1ce", "t2", "flair"]


def _load_nifti_volume(subject_id: str, data_root: Path) -> torch.Tensor:
    import SimpleITK as sitk

    subject_path = Path(data_root) / subject_id
    if not subject_path.exists():
        raise FileNotFoundError(
            f"Subject folder not found: {subject_path}\nExpected: {{data_root}}/{{subject_id}}/"
        )

    channels = []
    for modality in _CHANNEL_ORDER:
        filepath = subject_path / f"{subject_id}{_MODALITY_SUFFIXES[modality]}"
        if not filepath.exists():
            raise FileNotFoundError(f"Missing modality file: {filepath}")
        img = sitk.ReadImage(str(filepath))
        array = sitk.GetArrayFromImage(img).astype(np.float32)
        channels.append(array)

    volume = np.stack(channels, axis=0)
    transforms = get_val_transforms()
    sample = transforms({"image": volume, "label": torch.tensor(0)})
    tensor = sample["image"]
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    return tensor.unsqueeze(0).float()


def run_inference(subject_id: Optional[str] = None, gradcam_model: Optional[str] = None) -> Dict:
    if not subject_id:
        raise ValueError("Provide a subject_id string, e.g. --subject-id BraTS2021_00000")

    if subject_id.isdigit():
        subject_id = f"BraTS2021_{int(subject_id):05d}"

    tensor = _load_nifti_volume(subject_id, BRATS_SUBJECTS_DIR).to(DEVICE)
    ensemble = ReliabilityWeightedEnsemble()

    if not ensemble.models:
        raise RuntimeError("No trained model checkpoints found. Run train_models.py before inference.")

    result = ensemble.predict_tensor(tensor)
    result["subject_id"] = subject_id
    result["prediction"] = result["prediction"][0]
    result["probabilities"] = {
        "lgg": result["probabilities"][0][0],
        "hgg": result["probabilities"][0][1],
    }
    result["weights"] = {name: values[0] for name, values in result["weights"].items()}

    if gradcam_model:
        if gradcam_model not in ensemble.models:
            available = ", ".join(sorted(ensemble.models)) or "none"
            raise ValueError(f"Grad-CAM model '{gradcam_model}' is not loaded. Available: {available}")
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("imageio is required for Grad-CAM export. Install dependencies from requirements.txt.") from exc
        try:
            from .gradcam import generate_gradcam
        except ImportError:
            try:
                from gradcam import generate_gradcam
            except ImportError as exc:
                raise RuntimeError("Grad-CAM export requires optional visualization dependencies from requirements.txt.") from exc

        overlay = generate_gradcam(gradcam_model, ensemble.models[gradcam_model], tensor)
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
        out_path = GRADCAM_DIR / f"{subject_id}_{gradcam_model}_gradcam.png"
        imageio.imwrite(out_path, overlay)
        result["gradcam_path"] = str(out_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run 3D reliability-aware inference on a BraTS 2021 subject")
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="BraTS 2021 subject ID, e.g. BraTS2021_00000 or bare integer 0",
    )
    parser.add_argument("--gradcam-model", default=None, help="Model name to generate Grad-CAM overlay")
    args = parser.parse_args()

    output = run_inference(subject_id=args.subject_id, gradcam_model=args.gradcam_model)
    print(json.dumps(output, indent=2))
