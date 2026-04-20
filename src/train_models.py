"""
True 3D BraTS 2021 training pipeline for LGG-vs-HGG reliability-aware ensembling.

Migration notes:
- Removed the old slice/H5 manifest-based loader from the main training path.
- Added BraTS 2021 NIfTI subject-folder loading via the src utility modules.
- Added weighted CrossEntropyLoss computed from the training split only.
- Kept the 3D backbone builders, AMP, gradient accumulation, early stopping,
  uncertainty estimation, and registry flow.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE,
    BRATS_LABELS_CSV,
    BRATS_SUBJECTS_DIR,
    CLASSES,
    DEVICE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    INPUT_CHANNELS,
    INPUT_DEPTH,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    LEARNING_RATE,
    MC_DROPOUT_PASSES,
    MEDICALNET_WEIGHTS_PATH,
    MODELS,
    RANDOM_SEED,
    REGISTRY_PATH,
    RESULTS_DIR,
    SPATIAL_SIZE,
    TEST_RATIO,
    TRAIN_RATIO,
    USE_MIXED_PRECISION,
    VAL_RATIO,
    WEIGHT_DECAY,
    create_directories,
    get_model_path,
)

try:
    from .brats_dataset import build_dataloaders as build_nifti_dataloaders
    from .data_transforms import get_train_transforms, get_val_transforms
    from .label_loader import load_labels
    from .splits import make_splits, verify_no_leakage
except ImportError:
    from brats_dataset import build_dataloaders as build_nifti_dataloaders
    from data_transforms import get_train_transforms, get_val_transforms
    from label_loader import load_labels
    from splits import make_splits, verify_no_leakage


# ================================================================== #
# DataLoader factory                                                 #
# ================================================================== #

def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    label_col: str = "label_grade",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Build train / val / test DataLoaders from BraTS 2021 NIfTI subject folders.

    Returns:
        train_loader, val_loader, test_loader, class_weights
        class_weights: numpy array [w_LGG, w_HGG] computed from train split only
    """
    df = load_labels(csv_path=BRATS_LABELS_CSV, data_root=BRATS_SUBJECTS_DIR)

    df_train, df_val, df_test = make_splits(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        stratify_col=label_col,
        random_seed=RANDOM_SEED,
    )

    verify_no_leakage(df_train, df_val, df_test)

    train_labels = df_train[label_col].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )

    train_loader, val_loader, test_loader = build_nifti_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        train_transforms=get_train_transforms(),
        val_transforms=get_val_transforms(),
        batch_size=batch_size,
        num_workers=num_workers,
        label_col=label_col,
    )

    return train_loader, val_loader, test_loader, class_weights


# ================================================================== #
# Model builders                                                     #
# ================================================================== #
#
# Architectural diversity is the theoretical foundation of this ensemble.
# The failure mode of any ensemble fusion is correlated errors — samples
# where all models fail simultaneously. Heterogeneous inductive biases
# de-correlate errors by construction:
#
#   ResNet-18 + MedicalNet : local convolution, medical domain init
#   DenseNet-121           : dense multi-scale feature reuse
#   Swin Transformer 3D    : global window-attention across the volume
#
# These three architectures see the same 4-channel 3D volume through
# fundamentally different computational lenses. The reliability-weighted
# ensemble exploits this: when one model is uncertain, the others are
# structurally more likely to be confident and correct.
# ================================================================== #

def _get_monai_networks():
    try:
        from monai.networks import nets as monai_nets
    except ImportError as exc:
        raise ImportError(
            "MONAI is required. Install with: pip install monai"
        ) from exc
    return monai_nets


# ------------------------------------------------------------------ #
# Shared utility                                                      #
# ------------------------------------------------------------------ #

class BackboneWithDropoutHead(nn.Module):
    """
    Wraps a MONAI backbone: replaces the original classifier with
    an Identity, then applies Dropout3d → Linear.
    Works for any backbone that returns a 2D or spatial feature tensor.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head_attr: str,
        in_features: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        setattr(self.backbone, head_attr, nn.Identity())
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Flatten spatial dims if needed
        if features.ndim > 2:
            features = features.mean(dim=list(range(2, features.ndim)))
        return self.classifier(self.dropout(features))


# ------------------------------------------------------------------ #
# Model 1: ResNet50-3D with optional MedicalNet pretraining          #
# ------------------------------------------------------------------ #

def _try_load_medicalnet_weights(model: nn.Module, weights_path: str) -> bool:
    """
    Attempt to load MedicalNet pretrained weights into model.backbone.
    Key remapping handles the 'module.' prefix from DataParallel saves.

    Download weights from:
    https://github.com/Tencent/MedicalNet/releases (resnet_50_23dataset.pth)
    Place at: models/medicalnet_resnet50.pth

    Returns True if weights loaded successfully.
    """
    from pathlib import Path
    p = Path(weights_path)
    if not p.exists():
        print(f"[MedicalNet] Weights not found at {p}. Training from scratch.")
        return False

    try:
        ckpt = torch.load(str(p), map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Strip 'module.' prefix (DataParallel artefact)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        # MedicalNet uses 1-channel input; our model uses 4-channel.
        # Skip conv1 weight — it will be randomly initialized.
        state = {k: v for k, v in state.items() if "conv1" not in k}
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        print(f"[MedicalNet] Loaded weights from {p.name}")
        print(f"  Missing keys   : {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        return True
    except Exception as exc:
        print(f"[MedicalNet] Failed to load weights: {exc}. Training from scratch.")
        return False


def build_resnet18_medicalnet(num_classes: int = 2, in_channels: int = INPUT_CHANNELS) -> nn.Module:
    """
    3D ResNet-18 backbone with MedicalNet pretrained weights.
    MedicalNet was pretrained on 8 large 3D medical imaging datasets
    (CT + MRI), giving strong initialization for BraTS volumetric input.

    Weights: models/resnet_18_23dataset.pth  (126 MB, confirmed present)
    Falls back to random init if weights not found.
    """
    monai_nets = _get_monai_networks()
    backbone = monai_nets.resnet18(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=num_classes,
    )
    model = BackboneWithDropoutHead(
        backbone,
        "fc",
        backbone.fc.in_features,
        num_classes,
        MODELS["resnet18_medicalnet"]["dropout"],
    )
    if MEDICALNET_WEIGHTS_PATH:
        _try_load_medicalnet_weights(model, MEDICALNET_WEIGHTS_PATH)
    return model


# ------------------------------------------------------------------ #
# Model 2: DenseNet121-3D (unchanged — strong dense CNN baseline)    #
# ------------------------------------------------------------------ #

def build_densenet121_3d(num_classes: int = 2, in_channels: int = INPUT_CHANNELS) -> nn.Module:
    """
    3D DenseNet-121: dense skip connections enable strong feature reuse.
    Well-validated on BraTS and similar volumetric tasks.
    """
    monai_nets = _get_monai_networks()
    backbone = monai_nets.DenseNet121(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
    )
    # DenseNet121 classifier is backbone.class_layers
    # Get in_features from the Linear inside class_layers
    linear = next(
        (m for m in backbone.class_layers.modules() if isinstance(m, nn.Linear)),
        None,
    )
    if linear is None:
        raise RuntimeError("Could not find Linear layer in DenseNet121 class_layers.")
    return BackboneWithDropoutHead(
        backbone,
        "class_layers",
        linear.in_features,
        num_classes,
        MODELS["densenet121_3d"]["dropout"],
    )


# ------------------------------------------------------------------ #
# Model 3: Swin Transformer 3D — window-attention transformer        #
# ------------------------------------------------------------------ #

class SwinClassifier3D(nn.Module):
    """
    3D Swin Transformer classifier using MONAI's SwinUNETR encoder.

    Architecture:
      SwinUNETR encoder (feature_size=24, 4 stages of Swin windows)
      → bottleneck feature map (B, feature_size*16, d/32, h/32, w/32)
      → global average pooling  → (B, feature_size*16)
      → Dropout → Linear → num_classes logits

    For img_size=(96,96,96) [RTX A1000 8 GB config] and feature_size=24:
      Bottleneck: 384 channels, spatial 3×3×3
      Total classifier params: ~150 K
      Peak VRAM: ~5–6 GB with batch=1 + fp16 AMP ✔

    For img_size=(128,128,128) [16+ GB GPU] and feature_size=24:
      Bottleneck: 384 channels, spatial 4×4×4
      Peak VRAM: ~10–12 GB — OOM on 8 GB cards

    Grad-CAM target: swin_encoder.swinViT.layers4 (last Swin stage)
    """
    def __init__(
        self,
        img_size: tuple,
        in_channels: int,
        num_classes: int,
        feature_size: int,
        dropout: float,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.swin_encoder = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=num_classes,   # dummy — encoder only
            feature_size=feature_size,
            use_checkpoint=False,       # set True to save memory at training
        )
        bottleneck_channels = feature_size * 16   # 24*16 = 384
        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(bottleneck_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swinViT returns a list of 5 feature maps (enc0 … enc4 / bottleneck)
        hidden_states = self.swin_encoder.swinViT(x, normalize=self.swin_encoder.normalize)
        # Use the deepest (bottleneck) feature map — index 4
        bottleneck = hidden_states[4]              # (B, C, d, h, w)
        pooled     = self.pool(bottleneck).flatten(1)   # (B, C)
        return self.classifier(self.dropout(pooled))


def build_swin_transformer_3d(
    num_classes: int = 2,
    in_channels: int = INPUT_CHANNELS,
) -> nn.Module:
    """
    Swin Transformer 3D — window-based self-attention, captures long-range
    global context that CNNs miss.  Uses MONAI's SwinUNETR encoder backbone.

    feature_size=24 → 'Swin-Tiny' scale.
    Input: (B, 4, 128, 128, 128) — 4-channel BraTS 2021 3D volume.
    Use config SPATIAL_SIZE to control resolution; must be divisible by 32.
    """
    cfg = MODELS["swin_transformer_3d"]
    return SwinClassifier3D(
        img_size=SPATIAL_SIZE,
        in_channels=in_channels,
        num_classes=num_classes,
        feature_size=cfg["feature_size"],
        dropout=cfg["dropout"],
    )


# ------------------------------------------------------------------ #
# Factory                                                             #
# ------------------------------------------------------------------ #

def get_model(model_name: str) -> nn.Module:
    builders = {
        "resnet18_medicalnet":  build_resnet18_medicalnet,
        "densenet121_3d":       build_densenet121_3d,
        "swin_transformer_3d":  build_swin_transformer_3d,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(builders)}")
    return builders[model_name]()




# ================================================================== #
# Training utilities                                                 #
# ================================================================== #

class EarlyStopping:
    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, score: float, model: nn.Module) -> bool:
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
) -> Dict[str, float]:
    """
    Single training or validation epoch.
    Expects loaders that yield dicts with keys 'image' and 'label'.
    """
    is_train = optimizer is not None
    model.train(mode=is_train)
    losses: List[float] = []
    y_true: List[np.ndarray] = []
    y_prob: List[np.ndarray] = []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    amp_enabled = bool(USE_MIXED_PRECISION and DEVICE.type == "cuda")

    for step, batch in enumerate(tqdm(loader, leave=False), start=1):
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            logits = model(inputs)
            loss = criterion(logits, labels)
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS if is_train else loss

        if is_train:
            if scaler is not None and amp_enabled:
                scaler.scale(scaled_loss).backward()
                if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                scaled_loss.backward()
                if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        y_true.append(labels.detach().cpu().numpy())
        y_prob.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    true = np.concatenate(y_true)
    prob = np.concatenate(y_prob)
    pred = prob.argmax(axis=1)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(true, pred)),
        "f1": float(f1_score(true, pred, zero_division=0)),
        "auc": float(roc_auc_score(true, prob[:, 1])) if len(np.unique(true)) > 1 else 0.0,
    }


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def estimate_mean_uncertainty(
    model: nn.Module,
    loader: DataLoader,
    passes: int = MC_DROPOUT_PASSES,
) -> float:
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(DEVICE)
            samples = []
            enable_dropout(model)
            for _ in range(passes):
                samples.append(torch.softmax(model(inputs), dim=1))
            preds = torch.stack(samples, dim=0)
            scores.append(preds.var(dim=0).mean(dim=1).detach().cpu().numpy())
    return float(np.concatenate(scores).mean())


# ================================================================== #
# Registry                                                           #
# ================================================================== #

def _load_registry() -> Dict[str, Dict]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _update_registry(
    model_name: str,
    val_accuracy: float,
    mean_uncertainty: float,
    checkpoint_path: Path,
    label_col: str = "label_grade",
) -> None:
    registry = _load_registry()
    registry[model_name] = {
        "model_name":        model_name,
        "val_accuracy":      val_accuracy,
        "mean_uncertainty":  mean_uncertainty,
        "checkpoint_path":   str(checkpoint_path),
        "label_col":         label_col,        # which task this checkpoint was trained for
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


# ================================================================== #
# Main training loop                                                 #
# ================================================================== #

def train_model(
    model_name: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    label_col: str = "label_grade",
) -> Dict:
    train_loader, val_loader, _, class_weights = create_dataloaders(
        batch_size=batch_size,
        label_col=label_col,
    )

    model = get_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(
        device=DEVICE.type,
        enabled=bool(USE_MIXED_PRECISION and DEVICE.type == "cuda"),
    )
    stopper = EarlyStopping()
    history = []
    best_val_accuracy = 0.0
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, scaler)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, criterion, optimizer=None, scaler=None)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        best_val_accuracy = max(best_val_accuracy, val_metrics["accuracy"])

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if stopper.step(val_metrics["accuracy"], model):
            print(f"Early stopping at epoch {epoch}.")
            break

    stopper.restore(model)

    checkpoint = get_model_path(model_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_shape": [INPUT_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH],
            "classes": CLASSES,
            "history": history,
            "label_col": label_col,
            "class_weights": class_weights.tolist(),
        },
        checkpoint,
    )

    mean_uncertainty = estimate_mean_uncertainty(model, val_loader)
    _update_registry(model_name, best_val_accuracy, mean_uncertainty, checkpoint, label_col=label_col)

    result = {
        "model_name": model_name,
        "best_val_accuracy": best_val_accuracy,
        "mean_uncertainty": mean_uncertainty,
        "checkpoint_path": str(checkpoint),
        "epochs_trained": len(history),
        "training_time_seconds": time.time() - start,
        "history": history,
        "training_config": {
            "input_shape": [INPUT_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH],
            "batch_size": batch_size,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "mixed_precision": bool(USE_MIXED_PRECISION and DEVICE.type == "cuda"),
            "label_col": label_col,
            "class_weights": class_weights.tolist(),
        },
    }
    (RESULTS_DIR / f"{model_name}_history.json").write_text(json.dumps(result, indent=2))
    return result


def train_all_models(
    model_names: Optional[List[str]] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    label_col: str = "label_grade",
) -> Dict[str, Dict]:
    create_directories()
    names = model_names or list(MODELS.keys())
    summary = {}
    for model_name in names:
        print(f"\n{'=' * 60}")
        print(f"Training: {model_name}")
        print(f"{'=' * 60}")
        summary[model_name] = train_model(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            label_col=label_col,
        )
    (RESULTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train 3D BraTS 2021 LGG-vs-HGG models")
    parser.add_argument("--models", nargs="+", default=None, help="Model names to train (default: all)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--label-col",
        type=str,
        default="label_grade",
        choices=["label_grade", "label_idh"],
        help="Which label column to train on",
    )
    args = parser.parse_args()

    train_all_models(
        model_names=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        label_col=args.label_col,
    )
