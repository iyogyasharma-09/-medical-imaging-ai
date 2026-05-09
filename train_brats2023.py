"""
train_brats2023.py
------------------
Complete BraTS 2023 training pipeline with GPU + NPU support.

DEVICE STRATEGY
===============
  GPU (CUDA/NVIDIA RTX A1000 8GB)
    - Full 3D model training with AMP (fp16) + gradient accumulation
    - All weight updates happen here

  NPU (Intel AI Boost via OpenVINO)
    - NOT used for gradient-based training (NPUs are inference accelerators)
    - Used for post-training: export PyTorch → ONNX → OpenVINO IR
    - Run fast NPU inference (see export_to_npu.py)

  CPU fallback
    - Automatic fallback if CUDA unavailable

SUPPORTED MODELS
================
  resnet18_medicalnet  — 3D ResNet-18 + MedicalNet pretrain
  densenet121_3d       — 3D DenseNet-121
  swin_transformer_3d  — 3D Swin Transformer (window-attention)

USAGE
=====
  # Train all models on GPU (default)
  python train_brats2023.py

  # Train one model, custom epochs
  python train_brats2023.py --model resnet18_medicalnet --epochs 20

  # Train GLI only with grade labels (LGG vs HGG)
  python train_brats2023.py --task GLI --label-col label_grade

  # Train all tasks (GLI=0, MEN=1, PED=2) as 3-class problem
  python train_brats2023.py --task ALL --num-classes 3
"""

import argparse
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

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import (
    BATCH_SIZE,
    BRATS2023_TEST_MANIFEST,
    BRATS2023_TRAIN_MANIFEST,
    BRATS2023_VAL_MANIFEST,
    DEFAULT_DROPOUT,
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
    MODELS_DIR,
    NUM_WORKERS,
    RANDOM_SEED,
    REGISTRY_PATH,
    RESULTS_DIR,
    SPATIAL_SIZE,
    USE_MIXED_PRECISION,
    WEIGHT_DECAY,
    create_directories,
    get_model_path,
)
from imported_files.brats2023_dataset import BraTS2023Dataset, load_manifest
from imported_files.transforms import get_train_transforms, get_val_transforms


# ===========================================================================
# Device setup
# ===========================================================================

def get_training_device() -> torch.device:
    """Returns the best available PyTorch training device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        p   = torch.cuda.get_device_properties(0)
        print(f"[Device] GPU: {p.name}  ({p.total_memory/1024**3:.1f} GB VRAM)")
        return dev
    print("[Device] CUDA unavailable — falling back to CPU.")
    return torch.device("cpu")


# ===========================================================================
# DataLoader builder
# ===========================================================================

def build_dataloaders(
    task:       str = "GLI",
    label_col:  str = "label",
    batch_size: int = BATCH_SIZE,
    num_workers:int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Load manifests and build train/val/test DataLoaders.

    Args:
        task      : 'GLI' | 'MEN' | 'PED' | 'ALL'
                    Filters the manifest to the chosen task subset.
                    'ALL' uses all tasks (3-class problem).
        label_col : 'label'       → task-based (GLI=0 MEN=1 PED=2)
                    'label_grade' → LGG=0 HGG=1 (GLI only, requires grade mapping)
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    def _load_and_filter(path: Path) -> "pd.DataFrame":
        import pandas as pd
        df = load_manifest(path, label_col=label_col)
        if task != "ALL":
            df = df[df["task"] == task].copy().reset_index(drop=True)
        return df

    df_train = _load_and_filter(BRATS2023_TRAIN_MANIFEST)
    df_val   = _load_and_filter(BRATS2023_VAL_MANIFEST)
    df_test  = _load_and_filter(BRATS2023_TEST_MANIFEST)

    print(f"\n[Data] Task={task}  label_col={label_col}")
    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    # Class weights from training split only
    labels = df_train[label_col].values.astype(int)
    unique = np.unique(labels)
    class_weights = compute_class_weight("balanced", classes=unique, y=labels)
    # Pad to max_class+1 so indexing works for 3-class
    full_weights = np.ones(int(unique.max()) + 1)
    for cls, w in zip(unique, class_weights):
        full_weights[int(cls)] = w
    print(f"  Class weights: {dict(zip(unique.tolist(), class_weights.round(3).tolist()))}")

    train_ds = BraTS2023Dataset(df_train, transforms=get_train_transforms(), label_col=label_col)
    val_ds   = BraTS2023Dataset(df_val,   transforms=get_val_transforms(),   label_col=label_col)
    test_ds  = BraTS2023Dataset(df_test,  transforms=get_val_transforms(),   label_col=label_col)

    nw = NUM_WORKERS
    print(f"  num_workers      : {nw}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=bool(nw),
                              multiprocessing_context="spawn" if nw > 0 else None,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True,
                              persistent_workers=bool(nw),
                              multiprocessing_context="spawn" if nw > 0 else None)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False,
                              num_workers=0,  pin_memory=True)
    return train_loader, val_loader, test_loader, full_weights


# ===========================================================================
# Model builders (3D, BraTS-compatible)
# ===========================================================================

class BackboneWithDropoutHead(nn.Module):
    """Wraps any MONAI backbone with a Dropout → Linear classifier head."""
    def __init__(self, backbone, head_attr, in_features, num_classes, dropout):
        super().__init__()
        self.backbone = backbone
        setattr(self.backbone, head_attr, nn.Identity())
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = feat.mean(dim=list(range(2, feat.ndim)))
        return self.classifier(self.dropout(feat))


def build_resnet18(num_classes: int) -> nn.Module:
    from monai.networks import nets
    backbone = nets.resnet18(spatial_dims=3, n_input_channels=INPUT_CHANNELS,
                             num_classes=num_classes)
    model = BackboneWithDropoutHead(backbone, "fc", backbone.fc.in_features,
                                    num_classes, DEFAULT_DROPOUT)
    # Try MedicalNet weights
    if MEDICALNET_WEIGHTS_PATH:
        p = Path(MEDICALNET_WEIGHTS_PATH)
        if p.exists():
            ckpt  = torch.load(str(p), map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()
                     if "conv1" not in k}
            missing, _ = model.backbone.load_state_dict(state, strict=False)
            print(f"  [MedicalNet] Loaded weights ({len(missing)} missing keys)")
        else:
            print(f"  [MedicalNet] Weights not found — training from scratch")
    return model


def build_densenet121(num_classes: int) -> nn.Module:
    from monai.networks import nets
    backbone = nets.DenseNet121(spatial_dims=3, in_channels=INPUT_CHANNELS,
                                out_channels=num_classes)
    linear = next((m for m in backbone.class_layers.modules()
                   if isinstance(m, nn.Linear)), None)
    return BackboneWithDropoutHead(backbone, "class_layers", linear.in_features,
                                   num_classes, DEFAULT_DROPOUT)


class SwinClassifier3D(nn.Module):
    def __init__(self, num_classes, feature_size=24, dropout=DEFAULT_DROPOUT):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        # Updated MONAI SwinUNETR API: img_size removed, uses spatial_dims instead
        self.enc = SwinUNETR(
            in_channels=INPUT_CHANNELS,
            out_channels=num_classes,
            feature_size=feature_size,
            use_checkpoint=False,
            spatial_dims=3,
        )
        self.pool       = nn.AdaptiveAvgPool3d(1)
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_size * 16, num_classes)

    def forward(self, x):
        # swinViT returns (hidden_states_out, bottleneck) in newer MONAI
        out = self.enc.swinViT(x, normalize=self.enc.normalize)
        # Handle both old list-of-5 and new (hidden_states, bottleneck) return styles
        if isinstance(out, (list, tuple)) and len(out) == 2 and hasattr(out[1], 'shape'):
            bottleneck = out[1]   # new API: (hidden_states_out, bottleneck)
        else:
            bottleneck = out[4]   # old API: list of 5 feature maps, index 4 = deepest
        pooled = self.pool(bottleneck).flatten(1)
        return self.classifier(self.dropout(pooled))


def get_model(model_name: str, num_classes: int) -> nn.Module:
    builders = {
        "resnet18_medicalnet": build_resnet18,
        "densenet121_3d":      build_densenet121,
        "swin_transformer_3d": lambda n: SwinClassifier3D(n),
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model '{model_name}'. Options: {list(builders)}")
    print(f"  Building {model_name} (num_classes={num_classes})")
    return builders[model_name](num_classes)


# ===========================================================================
# Early stopping
# ===========================================================================

class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        improved = self.best_score is None or score > self.best_score + self.min_delta
        if improved:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ===========================================================================
# Training / evaluation epoch
# ===========================================================================

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              grad_accum=GRADIENT_ACCUMULATION_STEPS) -> Dict:
    is_train  = optimizer is not None
    amp_on    = bool(USE_MIXED_PRECISION and device.type == "cuda")
    model.train(mode=is_train)
    losses, y_true, y_prob = [], [], []

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, leave=False, desc="train" if is_train else "val"),
                                 start=1):
        imgs   = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=amp_on):
            logits = model(imgs)
            loss   = criterion(logits, labels)
            sloss  = loss / grad_accum if is_train else loss

        if is_train:
            if scaler and amp_on:
                scaler.scale(sloss).backward()
                if step % grad_accum == 0 or step == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                sloss.backward()
                if step % grad_accum == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        y_true.append(labels.detach().cpu().numpy())
        y_prob.append(torch.softmax(logits, dim=1).detach().cpu().numpy())

    true = np.concatenate(y_true)
    prob = np.concatenate(y_prob)
    pred = prob.argmax(axis=1)
    n_classes = prob.shape[1]

    metrics = {
        "loss":     float(np.mean(losses)),
        "accuracy": float(accuracy_score(true, pred)),
        "f1":       float(f1_score(true, pred, average="weighted", zero_division=0)),
    }
    # AUC: only if more than one class present and binary/OvR
    if len(np.unique(true)) > 1:
        try:
            avg  = "ovr" if n_classes > 2 else None
            auc  = roc_auc_score(true, prob if n_classes > 2 else prob[:, 1],
                                 multi_class=avg or "raise")
            metrics["auc"] = float(auc)
        except Exception:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0
    return metrics


# ===========================================================================
# MC-Dropout uncertainty
# ===========================================================================

def estimate_uncertainty(model, loader, device, passes=MC_DROPOUT_PASSES) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            # Activate dropout for uncertainty
            for m in model.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.train()
            preds = torch.stack([torch.softmax(model(imgs), 1) for _ in range(passes)])
            scores.append(preds.var(0).mean(1).cpu().numpy())
    return float(np.concatenate(scores).mean())


# ===========================================================================
# Registry
# ===========================================================================

def update_registry(model_name, val_acc, uncertainty, ckpt_path, task, label_col, device_used):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {}
    reg[model_name] = {
        "model_name":        model_name,
        "val_accuracy":      val_acc,
        "mean_uncertainty":  uncertainty,
        "checkpoint_path":   str(ckpt_path),
        "task":              task,
        "label_col":         label_col,
        "trained_on_device": device_used,
        "onnx_path":         str(ckpt_path).replace(".pt", ".onnx"),
        "openvino_path":     str(ckpt_path).replace(".pt", "_ov/model.xml"),
    }
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


# ===========================================================================
# Main training loop
# ===========================================================================

def train_model(
    model_name:  str,
    task:        str = "GLI",
    label_col:   str = "label_grade",
    num_classes: int = 2,
    epochs:      int = EPOCHS,
    batch_size:  int = BATCH_SIZE,
    device:      Optional[torch.device] = None,
    deadline:    Optional[float] = None,   # time.time() deadline; None = no limit
) -> Dict:
    if device is None:
        device = get_training_device()

    time_limit_str = "no limit" if deadline is None else f"{(deadline-time.time())/3600:.2f}h remaining"
    print(f"\n{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Task       : {task}  |  Label : {label_col}")
    print(f"  Classes    : {num_classes}  |  Device : {device}")
    print(f"  Epochs     : {epochs}  |  Batch : {batch_size}  |  Accum : {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  AMP (fp16) : {USE_MIXED_PRECISION and device.type == 'cuda'}")
    print(f"  Time limit : {time_limit_str}")
    print(f"{'='*60}")

    train_loader, val_loader, _, class_weights = build_dataloaders(
        task=task, label_col=label_col, batch_size=batch_size)

    # ---- Build model ----
    model     = get_model(model_name, num_classes).to(device)
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(device=device.type,
                                     enabled=bool(USE_MIXED_PRECISION and device.type == "cuda"))
    stopper   = EarlyStopping()
    history   = []
    best_acc  = 0.0
    t0        = time.time()

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history.append({"epoch": epoch, "train": train_m, "val": val_m,
                        "lr": optimizer.param_groups[0]["lr"]})
        best_acc = max(best_acc, val_m["accuracy"])

        print(f"Ep {epoch:3d}/{epochs} "
              f"| train loss={train_m['loss']:.4f} acc={train_m['accuracy']:.4f} "
              f"| val loss={val_m['loss']:.4f} acc={val_m['accuracy']:.4f} "
              f"auc={val_m['auc']:.4f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e}")

        if stopper.step(val_m["accuracy"], model):
            print(f"  [EarlyStopping] Triggered at epoch {epoch}.")
            break

        # Wall-clock time limit check
        if deadline is not None and time.time() >= deadline:
            remaining = 0.0
            print(f"  [TimeLimit] 2-hour wall-clock limit reached after epoch {epoch}. Saving checkpoint.")
            break

    stopper.restore(model)

    # ---- Save PyTorch checkpoint ----
    ckpt_path = get_model_path(model_name)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name":       model_name,
        "task":             task,
        "label_col":        label_col,
        "num_classes":      num_classes,
        "input_shape":      [INPUT_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH],
        "spatial_size":     list(SPATIAL_SIZE),
        "history":          history,
        "class_weights":    class_weights.tolist(),
        "best_val_accuracy": best_acc,
    }, ckpt_path)
    print(f"\n  [GPU] Checkpoint saved: {ckpt_path}")

    # ---- MC-Dropout uncertainty ----
    uncertainty = estimate_uncertainty(model, val_loader, device)
    update_registry(model_name, best_acc, uncertainty, ckpt_path,
                    task, label_col, str(device))

    # ---- Export to ONNX + OpenVINO for NPU ----
    print(f"\n  [NPU] Exporting model for NPU inference...")
    try:
        export_to_npu(model, model_name, num_classes, device)
    except Exception as e:
        print(f"  [NPU] Export warning: {e}  (training results still saved)")

    total_time = time.time() - t0
    result = {
        "model_name":           model_name,
        "task":                 task,
        "label_col":            label_col,
        "best_val_accuracy":    best_acc,
        "mean_uncertainty":     uncertainty,
        "checkpoint_path":      str(ckpt_path),
        "epochs_trained":       len(history),
        "training_time_seconds": total_time,
        "device":               str(device),
        "history":              history,
    }
    (RESULTS_DIR / f"{model_name}_history.json").write_text(json.dumps(result, indent=2))
    print(f"  Training complete in {total_time/60:.1f} min  |  Best val acc: {best_acc:.4f}")
    return result


# ===========================================================================
# NPU Export: PyTorch → ONNX → OpenVINO IR
# ===========================================================================

def export_to_npu(model: nn.Module, model_name: str, num_classes: int,
                  device: torch.device) -> Dict[str, str]:
    """
    Export a trained PyTorch model to:
      1. ONNX  (.onnx)  — portable, runs on any hardware via ONNX Runtime
      2. OpenVINO IR (.xml + .bin) — optimised for Intel GPU.0 and NPU

    The OpenVINO IR can be loaded and run on:
      - GPU.0 (Intel Arc iGPU)   — low-power inference
      - NPU   (Intel AI Boost)   — fastest low-power inference
      - GPU.1 (NVIDIA RTX A1000) — via OpenVINO
      - CPU   (fallback)

    Returns dict of exported paths.
    """
    import onnx
    import openvino as ov

    model.eval().cpu()
    dummy = torch.zeros(1, INPUT_CHANNELS, *SPATIAL_SIZE)

    # ---- Step 1: Export to ONNX ----
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}},
        opset_version=17,
    )
    # Validate
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"    ONNX exported: {onnx_path}")

    # ---- Step 2: Convert ONNX → OpenVINO IR ----
    ov_dir = MODELS_DIR / f"{model_name}_ov"
    ov_dir.mkdir(exist_ok=True)
    xml_path = ov_dir / "model.xml"

    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))

    # Compile and save as OpenVINO IR
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=True)
    print(f"    OpenVINO IR saved: {xml_path}")
    print(f"    Devices available for inference: {core.available_devices}")

    # Quick validation — compile for NPU
    try:
        compiled = core.compile_model(ov_model, "NPU")
        print(f"    [OK] Model compiled for NPU successfully")
        del compiled
    except Exception as e:
        print(f"    [WARN] NPU compile check: {e}")
        try:
            compiled = core.compile_model(ov_model, "GPU.0")
            print(f"    [OK] Model compiled for Intel iGPU (GPU.0) as fallback")
            del compiled
        except Exception as e2:
            print(f"    [WARN] GPU.0 fallback: {e2}")

    # Put model back on training device
    model.to(device)
    return {"onnx": str(onnx_path), "openvino_xml": str(xml_path)}


# ===========================================================================
# Train all models
# ===========================================================================

def train_all(
    model_names:      Optional[List[str]] = None,
    task:             str   = "GLI",
    label_col:        str   = "label_grade",
    num_classes:      int   = 2,
    epochs:           int   = EPOCHS,
    batch_size:       int   = BATCH_SIZE,
    time_limit_hours: float = 0.0,   # 0 = no limit
) -> Dict:
    create_directories()
    device  = get_training_device()
    names   = model_names or list(MODELS.keys())
    summary = {}

    # Shared wall-clock deadline across ALL models
    deadline = (time.time() + time_limit_hours * 3600) if time_limit_hours > 0 else None
    if deadline:
        print(f"[TimeLimit] Training will stop at {time_limit_hours:.1f}h from now "
              f"(deadline shared across {len(names)} models).")

    for name in names:
        if deadline is not None and time.time() >= deadline:
            print(f"[TimeLimit] No time left for {name}. Skipping.")
            break
        summary[name] = train_model(
            model_name=name, task=task, label_col=label_col,
            num_classes=num_classes, epochs=epochs,
            batch_size=batch_size, device=device,
            deadline=deadline,
        )

    (RESULTS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[DONE] Training complete. Summary -> {RESULTS_DIR}/training_summary.json")
    return summary


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BraTS 2023 GPU+NPU Training Pipeline")

    parser.add_argument("--model", nargs="+", default=None,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Model(s) to train (default: all)")
    parser.add_argument("--task", default="GLI",
                        choices=["GLI", "MEN", "PED", "ALL"],
                        help="Dataset task filter (default: GLI)")
    parser.add_argument("--label-col", default="label_grade",
                        choices=["label", "label_grade"],
                        help="Label column (label_grade=LGG/HGG, label=task-based)")
    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of output classes (2=binary, 3=multi-task)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--time-limit-hours", type=float, default=0.0,
                        help="Stop all training after this many hours (0 = no limit). "
                             "Deadline is shared across all models; best checkpoint is always saved.")

    args = parser.parse_args()

    # Auto-set num_classes if ALL task
    if args.task == "ALL" and args.num_classes == 2:
        args.num_classes = 3
        args.label_col   = "label"
        print("[Auto] task=ALL -> num_classes=3, label_col=label")

    model_names = None if args.model is None or args.model == ["all"] else args.model

    train_all(
        model_names=model_names,
        task=args.task,
        label_col=args.label_col,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        time_limit_hours=args.time_limit_hours,
    )
