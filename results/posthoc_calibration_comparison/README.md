# Post-Hoc Temperature Scaling Comparison Experiment

## Overview
This experiment evaluates post-hoc temperature scaling across all 9 canonical BraTS 2023 GLI grading checkpoints (Protocols A, B, C across ResNet18, DenseNet121, SwinUNETR) plus the Focal Loss ResNet18 intervention.

## Strict Leakage Controls
- **Validation-Only Fitting:** Temperature parameter $T^*$ was optimized strictly by minimizing Negative Log-Likelihood (NLL) on the 187 validation subjects.
- **Locked Test Isolation:** The 188 locked test subjects were processed strictly in read-only forward mode using the frozen $T^*$ parameters.
- **Mathematical Invariance:** Verified that relative rank orderings, ROC-AUC, and binary classifications at $\tau=0.50$ remain invariant under scalar temperature scaling.

## Cohort Details
- **Validation Split:** 187 subjects (10 LGG, 177 HGG) from `processed_dataset/brats2023/val_manifest.json`
- **Locked Test Split:** 188 subjects (12 LGG, 176 HGG) from `processed_dataset/brats2023/test_manifest.json`
- **Deterministic Transforms:** Standard `get_val_transforms()` ($4 \times 96 \times 96 \times 96$ volume normalization).

## Generated Files
- `multi_model_calibration_summary.csv`: Master metrics comparison table.
- `temperature_parameters.json`: Optimized frozen $T^*$ parameters and convergence metadata.
- `calibrated_predictions/`: Per-model subject-level calibrated predictions CSVs.
- `multi_model_reliability_diagrams.png`: 9-panel publication reliability diagram.
- `multi_model_threshold_comparison.csv`: Full threshold sweep table across all calibrated models.
- `experiment_metadata.json`: Complete machine-readable experiment manifest.
