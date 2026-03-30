"""
MC Dropout uncertainty estimation and reliability-weighted ensemble fusion for 3D models.

Uncertainty normalization (CRITICAL for fair ensemble weighting):
  Raw MC-Dropout variance scales differ across models due to architecture and
  dropout rate differences. Without normalization, one model's raw variance can
  dominate the ensemble weights regardless of actual accuracy.

  We apply: Ui_norm = Ui / max(Ui_model)  (per-model max normalization)
  Then:      Ri = (1 - Ui_norm) * Ai
  This ensures reliability scores are computed on a [0, 1] scale per model.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASSES, DEVICE, MC_DROPOUT_PASSES, MODELS, REGISTRY_PATH, get_model_path

try:
    from .train_models import enable_dropout, get_model
except ImportError:
    from train_models import enable_dropout, get_model


def compute_ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies  = (predictions == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (confidences > start) & (confidences <= end)
        if mask.any():
            ece += np.abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)


def normalize_uncertainty(uncertainty: np.ndarray) -> np.ndarray:
    """
    Per-model uncertainty normalization.
    Maps raw MC-Dropout variance to [0, 1] by dividing by its maximum.

    Purpose: different backbones (ResNet vs DenseNet vs Swin) have different
    uncertainty scales. Without this step, whichever model has the highest
    raw variance will always receive under-weighted reliability scores,
    biasing the ensemble toward lower-variance models regardless of accuracy.

    Args:
        uncertainty: (N,) array of per-sample MC variance from one model

    Returns:
        (N,) array of normalized uncertainty in [0, 1]
    """
    u_max = float(uncertainty.max())
    if u_max < 1e-10:
        return np.zeros_like(uncertainty)        # uniform certainty → all zeros
    return uncertainty / u_max


@dataclass
class MCPrediction:
    mean_probabilities:   np.ndarray   # (N, C) softmax-averaged over MC passes
    uncertainty_per_sample: np.ndarray # (N,)   per-sample MC variance (raw)

    @property
    def uncertainty(self) -> float:
        return float(self.uncertainty_per_sample.mean())


def mc_dropout_predict(model: nn.Module, inputs: torch.Tensor, passes: int = MC_DROPOUT_PASSES) -> MCPrediction:
    """Run MC Dropout inference: enable dropout, run `passes` forward passes, aggregate."""
    model.eval()
    enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(inputs)
            samples.append(torch.softmax(logits, dim=1))
    stacked          = torch.stack(samples, dim=0)          # (passes, N, C)
    mean_prediction  = stacked.mean(dim=0)                  # (N, C)
    variance         = stacked.var(dim=0).mean(dim=1)       # (N,) — mean over classes
    return MCPrediction(
        mean_probabilities    = mean_prediction.detach().cpu().numpy(),
        uncertainty_per_sample= variance.detach().cpu().numpy(),
    )


class ModelRegistry:
    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path

    def load(self) -> Dict[str, Dict]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}

    def get(self, model_name: str) -> Optional[Dict]:
        return self.load().get(model_name)


class ReliabilityWeightedEnsemble:
    """
    Reliability-Weighted Ensemble Fusion (RWEF).

    For each model i:
        Ui_norm = normalize_uncertainty(Ui)        # scale to [0,1] per model
        Ri      = (1 - Ui_norm) * Ai              # reliability score
        wi      = Ri / sum(Ri)                    # normalized weight
    Final:
        P_final = sum(wi * Pi)

    The per-model normalization step is critical: without it, models with
    architecturally higher variance (e.g. Swin Transformer with many dropout
    layers) would receive systematically lower weights irrespective of accuracy.

    --- Methodological justification for heterogeneous inductive biases ---
    The failure mode of any ensemble fusion method is correlated errors:
    when all constituent models are wrong on the same sample, no weighting
    scheme can recover a correct prediction. Architectural diversity — using
    a local convolutional network (ResNet-18 + MedicalNet), a densely
    connected CNN (DenseNet-121), and a window-attention transformer (Swin)
    — directly reduces this failure mode. By construction, these architectures
    have different inductive biases and therefore different error distributions:
    the CNN excels at local texture, DenseNet at multi-scale feature reuse,
    and Swin at long-range spatial relationships across the 3D volume. Their
    errors are structurally de-correlated, making the reliability-weighted
    fusion a principled design choice rather than an arbitrary model selection.
    This is the key claim that distinguishes weighted ensemble fusion from
    simple CNNensemble averaging.
    -----------------------------------------------------------------------
    """

    def __init__(self, model_names: Optional[List[str]] = None):
        self.model_names = model_names or list(MODELS.keys())
        self.registry    = ModelRegistry()
        self.models: Dict[str, nn.Module] = {}

        for model_name in self.model_names:
            info            = self.registry.get(model_name)
            checkpoint_path = Path(info["checkpoint_path"]) if info else get_model_path(model_name)
            if checkpoint_path.exists():
                model = get_model(model_name).to(DEVICE)
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self.models[model_name] = model

    def predict_tensor(self, inputs: torch.Tensor) -> Dict:
        if not self.models:
            raise RuntimeError(
                "No trained 3D model checkpoints found. Run train.py first."
            )
        batch_size = inputs.shape[0]
        per_model:        Dict[str, Dict]      = {}
        raw_uncertainties: Dict[str, np.ndarray] = {}
        mc_predictions:    Dict[str, MCPrediction] = {}

        # ── Step 1: MC Dropout inference per model ──────────────────────────
        for model_name, model in self.models.items():
            mc_pred = mc_dropout_predict(model, inputs.to(DEVICE))
            mc_predictions[model_name]    = mc_pred
            raw_uncertainties[model_name] = mc_pred.uncertainty_per_sample

        # ── Step 2: Per-model uncertainty normalization ──────────────────────
        # IMPORTANT: normalize within each model's own uncertainty distribution.
        # This prevents architecture-level variance differences from biasing weights.
        normalized_uncertainties = {
            name: normalize_uncertainty(u)
            for name, u in raw_uncertainties.items()
        }

        # ── Step 3: Reliability scores and weighted fusion ──────────────────
        reliability_scores: Dict[str, np.ndarray] = {}
        weighted_sum = np.zeros((batch_size, len(CLASSES)), dtype=np.float32)

        for model_name, model in self.models.items():
            registry_entry = self.registry.get(model_name) or {}
            accuracy       = float(registry_entry.get("val_accuracy", 0.5))
            mc_pred        = mc_predictions[model_name]
            ui_norm        = normalized_uncertainties[model_name]

            # Ri = (1 - Ui_norm) * Ai   — all values now in [0, 1] scale
            reliability = np.clip((1.0 - ui_norm) * accuracy, a_min=0.0, a_max=None)
            reliability_scores[model_name] = reliability
            weighted_sum += mc_pred.mean_probabilities * reliability[:, None]

            per_model[model_name] = {
                "probabilities":      mc_pred.mean_probabilities.tolist(),
                "raw_uncertainty":    raw_uncertainties[model_name].tolist(),
                "norm_uncertainty":   ui_norm.tolist(),
                "mean_uncertainty":   float(raw_uncertainties[model_name].mean()),
                "val_accuracy":       accuracy,
            }

        # ── Step 4: Normalize final weights ─────────────────────────────────
        total = np.sum(np.stack(list(reliability_scores.values()), axis=0), axis=0)
        total = np.where(total <= 0, 1.0, total)
        final_probabilities = weighted_sum / total[:, None]
        labels              = np.argmax(final_probabilities, axis=1)
        normalized_weights  = {
            name: (scores / total).tolist()
            for name, scores in reliability_scores.items()
        }

        return {
            "prediction":    [CLASSES[idx] for idx in labels],
            "probabilities": final_probabilities.tolist(),
            "weights":       normalized_weights,
            "model_details": per_model,
        }
