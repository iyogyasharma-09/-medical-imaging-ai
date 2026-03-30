"""
Central configuration for the BraTS LGG-vs-HGG 3D reliability-aware ensemble.

The project is in a migration state:
- Preferred path: BraTS 2021 subject-folder NIfTI pipeline
- Legacy compatibility path: older slice-based metadata/H5 pipeline
"""

from pathlib import Path

import torch

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
BRATS_DIR = DATASET_DIR / "brats"

# Preferred BraTS 2021 subject-folder dataset layout.
BRATS_SUBJECTS_DIR = BRATS_DIR / "BraTS2021_Training_Data"
BRATS_LABELS_CSV = BRATS_DIR / "BraTS2021_Training_Data_Labels.csv"

# Legacy compatibility paths kept for modules that still use the older slice/H5 flow.
BRATS_RAW_DIR = BRATS_DIR / "BraTS2021_training_data" / "content" / "data"
BRATS_METADATA_CSV = BRATS_DIR / "BraTS21 Training Metadata.csv"
BRATS_NAME_MAPPING_CSV = BRATS_RAW_DIR / "name_mapping.csv"

SPLIT_DIR = BASE_DIR / "processed_dataset" / "brats_lgg_hgg_3d"
TRAIN_MANIFEST = SPLIT_DIR / "train_manifest.json"
VAL_MANIFEST = SPLIT_DIR / "val_manifest.json"
TEST_MANIFEST = SPLIT_DIR / "test_manifest.json"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
REGISTRY_PATH = RESULTS_DIR / "model_registry.json"
EVALUATION_PATH = RESULTS_DIR / "evaluation_summary.json"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

CLASSES = ["lgg", "hgg"]
NUM_CLASSES = len(CLASSES)
TARGET_TO_CLASS = {0: "lgg", 1: "hgg"}
CLASS_TO_TARGET = {name: idx for idx, name in TARGET_TO_CLASS.items()}

MODALITIES = ["t1", "t1ce", "t2", "flair"]
MODALITY_TO_INDEX = {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3}
INPUT_CHANNELS = len(MODALITIES)
INPUT_DEPTH = 96
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
SPATIAL_SIZE = (INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH)
MC_DROPOUT_PASSES = 10
DEFAULT_DROPOUT = 0.4

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
MIN_NONZERO_FRAC = 0.10
MIN_INFORMATIVE_SLICES = 8

BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 0.0
GRADIENT_ACCUMULATION_STEPS = 2
USE_MIXED_PRECISION = True

# Model registry
#   1. resnet18_medicalnet — MONAI ResNet18 3D + MedicalNet pretrained weights
#   2. densenet121_3d      — dense CNN, strong feature reuse
#   3. swin_transformer_3d — Swin window-attention transformer (MONAI)
MODELS = {
    "resnet18_medicalnet":  {"dropout": DEFAULT_DROPOUT},
    "densenet121_3d":       {"dropout": DEFAULT_DROPOUT},
    "swin_transformer_3d": {"dropout": DEFAULT_DROPOUT, "feature_size": 24},
}

# MedicalNet pretrained ResNet-18 weights (confirmed present at this path)
MEDICALNET_WEIGHTS_PATH: str | None = str(MODELS_DIR / "resnet_18_23dataset.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_directories() -> None:
    for path in [SPLIT_DIR, MODELS_DIR, RESULTS_DIR, GRADCAM_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_model_path(model_name: str) -> Path:
    return MODELS_DIR / f"{model_name}_lgg_hgg_3d.pt"


if __name__ == "__main__":
    create_directories()
    print(f"Base directory: {BASE_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASSES}")
    print(f"Input shape: ({INPUT_CHANNELS}, {INPUT_DEPTH}, {INPUT_HEIGHT}, {INPUT_WIDTH})")
