# Day 7 — Audit 1: Protocol A/B/C Complete Forensic Provenance Audit

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading  
**Audit Date**: 2026-08-19  
**Audit Mode**: Strict Read-Only Forensic Provenance Audit  
**Auditor**: DeepMind Antigravity Advanced Agentic AI  
**Scope**: All 9 Canonical Configurations across Protocol A, Protocol B, and Protocol C  

---

## 1. Executive Summary & Forensic Verdicts

### 1.1 Summary of Findings
A strict, read-only forensic provenance audit was performed across all nine canonical experimental configurations in the BraTS 2023 Adult Glioma 3D grading project. The audit inspected actual checkpoint binaries, training histories, execution logs, dataset split manifests, preprocessing transformations, data loaders, locked-test prediction files, confusion matrices, and metric reports.

### 1.2 Verdict on Critical Comparability Question
**VERDICT: VALID CONTROLLED COMPARISON**  
Protocols A, B, and C constitute a **rigorously controlled experimental comparison** where class-weighting configuration is the sole intended independent variable:
- **Protocol A**: Standard unweighted CrossEntropyLoss ($w = \text{None}$, effective sum = 2.0).
- **Protocol B**: Inverse-frequency class weighting ($w = [1.9018, 0.0982]$, normalized to sum = 2.0).
- **Protocol C**: Minority-emphasized weighting with minority multiplier $\times 1.25$ ($w = [1.9207, 0.0793]$, normalized to sum = 2.0).

All other experimental dimensions—including dataset splits (Train $N=876$, Val $N=187$, Test $N=188$), random seed (`RANDOM_SEED = 42`), model architectures, weight initialization sources (MedicalNet for ResNet18; deterministic random initialization for DenseNet121 and SwinUNETR), spatial input resolution ($96 \times 96 \times 96$), preprocessing pipelines, augmentation routines, optimizer (`AdamW`, $\text{lr}=10^{-4}$, $\text{weight\_decay}=10^{-4}$), learning rate schedule (`CosineAnnealingLR`, $T_{\max}=30$), batch size ($2$), and evaluation decision threshold ($0.5$)—were held strictly constant. The variation in total epochs trained across configurations (7 to 25 epochs) is the direct, intended outcome of deterministic early stopping applied to distinct loss trajectories, not an uncontrolled experimental confounder.

### 1.3 Verdict on Critical Provenance Question
**VERDICT: GENUINELY DESIGNATED BY PROCEDURE**  
All nine archived Protocol A/B/C checkpoints are **genuinely and unambiguously the exact checkpoints designated by the project's documented intra-run training and early-stopping selection procedure**. They were NOT selected retrospectively based on locked-test performance:
1. **Deterministic Intra-Run Selection**: The training pipeline evaluates validation metrics after every epoch and maintains an intra-run best model buffer via `EarlyStopping(patience=6, tol=1e-4)`.
2. **Weight Restoration Verification**: In every single checkpoint, the saved `model_state_dict` corresponds exactly to the epoch where validation Macro-F1 (with LGG Recall as tie-breaker) achieved its intra-run maximum.
3. **Temporal Provenance**: Training logs confirm that each checkpoint was saved to disk upon training completion on Day 2 / Day 4, prior to the Day 5 Phase 1 locked-test evaluation run.

### 1.4 Integrity Confirmation
- **Protocol A Modifications**: 0 (0 expected)
- **Protocol B Modifications**: 0 (0 expected)
- **Protocol C Modifications**: 0 (0 expected)
- **Locked-Test Modifications**: 0 (0 expected)
- **Total Unexpected Modifications**: **0** (All SHA-256 hashes verified unchanged)

---

## 2. Inventory of the Nine Canonical Configurations

| ID | Protocol | Architecture | Checkpoint Filename | Size (Bytes) | Checkpoint SHA-256 |
|---|---|---|---|---|---|
| 1 | **Protocol A** | ResNet18 MedicalNet | `resnet18_medicalnet_lgg_hgg_3d.pt` | 133,002,639 | `4c9da69f2bf7bd7fd767cec41482f9f80f82241c92c2d4147ecfc2f4628dc631` |
| 2 | **Protocol A** | DenseNet121 3D | `densenet121_3d_lgg_hgg_3d.pt` | 45,885,866 | `b74310952c26ab95ada0a43cc285812d8a85693823efc8b5eab446926a956f75` |
| 3 | **Protocol A** | SwinUNETR 3D | `swin_transformer_3d_lgg_hgg_3d.pt` | 70,430,659 | `9902d819e3d38eaa17170ac85b286c40c470cfde827d80d3c935e36d3ff06f63` |
| 4 | **Protocol B** | ResNet18 MedicalNet | `resnet18_medicalnet_lgg_hgg_3d.pt` | 132,999,695 | `c30fdf777b06dfeb01be50747d6592fc645d071f47cda28febdd695f3fa24dd1` |
| 5 | **Protocol B** | DenseNet121 3D | `densenet121_3d_lgg_hgg_3d.pt` | 45,885,930 | `bfb7792111af196863a827e2e81bb914520a2870c52c75650db3389cdcd45a67` |
| 6 | **Protocol B** | SwinUNETR 3D | `swin_transformer_3d_lgg_hgg_3d.pt` | 70,430,723 | `bba4f1edaf72182059a7ed572d99b4098cd283bc27c0815bd5c5b67edec4b4b1` |
| 7 | **Protocol C** | ResNet18 MedicalNet | `resnet18_medicalnet_lgg_hgg_3d.pt` | 132,999,695 | `3d5b2989204f35f7c809628f71337dca629b1c1d83327e0d2e5c5768ed103160` |
| 8 | **Protocol C** | DenseNet121 3D | `densenet121_3d_lgg_hgg_3d.pt` | 45,885,930 | `b519a8acac0f1d54b990a4d08e374e1cebda8470794d50fdb0c54f0089cf7ffa` |
| 9 | **Protocol C** | SwinUNETR 3D | `swin_transformer_3d_lgg_hgg_3d.pt` | 70,432,451 | `2346cda6d0563779279b8b53ba43586c3a2a6a392120a5cc84b8eb48e510a0ff` |

---

## 3. Configuration-by-Configuration Provenance Profiles

### 3.1 Configuration 1: Protocol A — ResNet18 MedicalNet
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_A/resnet18_medicalnet_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `resnet18_medicalnet_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `4c9da69f2bf7bd7fd767cec41482f9f80f82241c92c2d4147ecfc2f4628dc631`
- **File Size**: 133,002,639 bytes
- **Epochs Trained**: 25
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 19**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6 epochs without validation improvement; triggers at Epoch 25, restoring weights from Epoch 19.
- **Loss Function**: `nn.CrossEntropyLoss(weight=None)` (Standard unweighted CrossEntropyLoss).
- **Exact Class-Weight Configuration**: `None` (Metadata: `class_weights: None`).
- **Random Seed**: `42` (Explicitly seeded in Python, NumPy, PyTorch CPU & CUDA).
- **Training Dataset / Split Source**: `processed_dataset/brats2023/train_manifest.json` (GLI filtered: $N=876$, 43 LGG, 833 HGG).
- **Validation Dataset / Split Source**: `processed_dataset/brats2023/val_manifest.json` (GLI filtered: $N=187$, 10 LGG, 177 HGG).
- **Locked Test Dataset Source**: `processed_dataset/brats2023/test_manifest.json` (GLI filtered: $N=188$, 12 LGG, 176 HGG).
- **Preprocessing Configuration**: MONAI dictionary transforms: `Orientationd(axcodes="RAS")`, `Spacingd(pixdim=(1.0, 1.0, 1.0), mode="bilinear")`, `CropForegroundd(select_fn=_select_nonzero)`, `Resized(spatial_size=(96, 96, 96), mode="trilinear")`, `NormalizeIntensityd(nonzero=True, channel_wise=True)`.
- **Augmentation Configuration**: `RandFlipd(prob=0.5, spatial_axis=0,1,2)`, `RandRotate90d(prob=0.5, max_k=3, axes=(0,1))`, `RandZoomd(prob=0.3, min_zoom=0.9, max_zoom=1.1)`, `RandShiftIntensityd(offsets=0.1, prob=0.5)`, `RandScaleIntensityd(factors=0.1, prob=0.5)`, `RandGaussianNoised(prob=0.2, std=0.05)`.
- **Model Initialization / Pretraining Source**: MONAI `resnet18` initialized with MedicalNet 23-dataset transfer learning weights (`models/resnet_18_23dataset.pth`), classification head replaced by `nn.Dropout(p=0.4)` + `nn.Linear(512, 2)`.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 19)**:
  - Macro-F1: **0.7642**
  - LGG Recall: **0.7000** (7/10)
  - Accuracy: **0.9412**
  - AUC: **0.9288**
  - Loss: **0.1475**
- **Locked-Test Prediction File**: `results/locked_test/protocol_A/resnet18_predictions.csv` (SHA-256: `5bb4bdf666aca237f76fef62ed9d1687685947fb12d032ed69e619054c3ac3ab`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9734**
  - Macro-F1: **0.8842**
  - LGG Recall: **0.7500** (9/12)
  - HGG Recall: **0.9886** (174/176)
  - AUC: **0.9830**
  - Precision: **0.9831**
  - Weighted-F1: **0.9748**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=9 & \text{FP}=3 \\ \text{FN}=2 & \text{TP}=174 \end{bmatrix}$

---

### 3.2 Configuration 2: Protocol A — DenseNet121 3D
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_A/densenet121_3d_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `densenet121_3d_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `b74310952c26ab95ada0a43cc285812d8a85693823efc8b5eab446926a956f75`
- **File Size**: 45,885,866 bytes
- **Epochs Trained**: 11
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 5**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 11, restoring weights from Epoch 5.
- **Loss Function**: `nn.CrossEntropyLoss(weight=None)`.
- **Exact Class-Weight Configuration**: `None` (Metadata: `class_weights: None`).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `DenseNet121(spatial_dims=3, in_channels=4, out_channels=2)` initialized randomly from scratch, classification head with `Dropout(p=0.4)` + `Linear(1024, 2)`.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 5)**:
  - Macro-F1: **0.6785**
  - LGG Recall: **0.5000** (5/10)
  - Accuracy: **0.9198**
  - AUC: **0.9339**
  - Loss: **0.1565**
- **Locked-Test Prediction File**: `results/locked_test/protocol_A/densenet121_predictions.csv` (SHA-256: `051c64ba7ab4a3716eef542eecc58d62c980d6abe1f5983e5c058cdd48c8730d`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9468**
  - Macro-F1: **0.7934**
  - LGG Recall: **0.6667** (8/12)
  - HGG Recall: **0.9659** (170/176)
  - AUC: **0.9422**
  - Precision: **0.9770**
  - Weighted-F1: **0.9497**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=8 & \text{FP}=4 \\ \text{FN}=6 & \text{TP}=170 \end{bmatrix}$

---

### 3.3 Configuration 3: Protocol A — SwinUNETR 3D
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_A/swin_transformer_3d_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `swin_transformer_3d_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `9902d819e3d38eaa17170ac85b286c40c470cfde827d80d3c935e36d3ff06f63`
- **File Size**: 70,430,659 bytes
- **Epochs Trained**: 7
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 1**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 7, restoring weights from Epoch 1.
- **Loss Function**: `nn.CrossEntropyLoss(weight=None)`.
- **Exact Class-Weight Configuration**: `None` (Metadata: `class_weights: None`).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `SwinUNETR(feature_size=24, spatial_dims=3)` initialized randomly from scratch, pooling via `AdaptiveAvgPool3d(1)` + `Dropout(p=0.4)` + `Linear(384, 2)`.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 1)**:
  - Macro-F1: **0.4863**
  - LGG Recall: **0.0000** (0/10)
  - Accuracy: **0.9465**
  - AUC: **0.8647**
  - Loss: **0.2131**
- **Locked-Test Prediction File**: `results/locked_test/protocol_A/swin_predictions.csv` (SHA-256: `f232d43c3bbe2c9d233dd80bd8fc26bb43c170e456b63a3ea674f2ecdaeb694d`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9362**
  - Macro-F1: **0.4835**
  - LGG Recall: **0.0000** (0/12)
  - HGG Recall: **1.0000** (176/176)
  - AUC: **0.9337**
  - Precision: **0.9362**
  - Weighted-F1: **0.9055**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=0 & \text{FP}=12 \\ \text{FN}=0 & \text{TP}=176 \end{bmatrix}$

---

### 3.4 Configuration 4: Protocol B — ResNet18 MedicalNet
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_B/resnet18_medicalnet_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `resnet18_medicalnet_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `c30fdf777b06dfeb01be50747d6592fc645d071f47cda28febdd695f3fa24dd1`
- **File Size**: 132,999,695 bytes
- **Epochs Trained**: 13
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 7**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 13, restoring weights from Epoch 7.
- **Loss Function**: `nn.CrossEntropyLoss(weight=cw_tensor)`.
- **Exact Class-Weight Configuration**: Inverse-frequency weights: $w_0 = 1.9018265$, $w_1 = 0.0981735$ (sum = 2.0).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `resnet18` initialized with MedicalNet 23-dataset weights.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 7)**:
  - Macro-F1: **0.7101**
  - LGG Recall: **0.7000** (7/10)
  - Accuracy: **0.9144**
  - AUC: **0.8929**
  - Loss: **0.2784**
- **Locked-Test Prediction File**: `results/locked_test/protocol_B/resnet18_predictions.csv` (SHA-256: `7c5f45c1f8beb3486b2fbbcc119773ba3ce385082a0cc0db30e4a257be03a099`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9149**
  - Macro-F1: **0.7267**
  - LGG Recall: **0.6667** (8/12)
  - HGG Recall: **0.9318** (164/176)
  - AUC: **0.8939**
  - Precision: **0.9762**
  - Weighted-F1: **0.9255**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=8 & \text{FP}=4 \\ \text{FN}=12 & \text{TP}=164 \end{bmatrix}$

---

### 3.5 Configuration 5: Protocol B — DenseNet121 3D
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_B/densenet121_3d_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `densenet121_3d_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `bfb7792111af196863a827e2e81bb914520a2870c52c75650db3389cdcd45a67`
- **File Size**: 45,885,930 bytes
- **Epochs Trained**: 11
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 5**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 11, restoring weights from Epoch 5.
- **Loss Function**: `nn.CrossEntropyLoss(weight=cw_tensor)`.
- **Exact Class-Weight Configuration**: Inverse-frequency weights: $w_0 = 1.9018265$, $w_1 = 0.0981735$ (sum = 2.0).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `DenseNet121` from scratch.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 5)**:
  - Macro-F1: **0.6785**
  - LGG Recall: **0.5000** (5/10)
  - Accuracy: **0.9198**
  - AUC: **0.8647**
  - Loss: **0.2482**
- **Locked-Test Prediction File**: `results/locked_test/protocol_B/densenet121_predictions.csv` (SHA-256: `d6f99a6c141748b8566c91123a0b53fc2e77115d497f59f285466ab3ccf11734`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9149**
  - Macro-F1: **0.7102**
  - LGG Recall: **0.5833** (7/12)
  - HGG Recall: **0.9375** (165/176)
  - AUC: **0.9460**
  - Precision: **0.9706**
  - Weighted-F1: **0.9238**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=7 & \text{FP}=5 \\ \text{FN}=11 & \text{TP}=165 \end{bmatrix}$

---

### 3.6 Configuration 6: Protocol B — SwinUNETR 3D
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_B/swin_transformer_3d_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `swin_transformer_3d_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `bba4f1edaf72182059a7ed572d99b4098cd283bc27c0815bd5c5b67edec4b4b1`
- **File Size**: 70,430,723 bytes
- **Epochs Trained**: 7
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 1**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 7, restoring weights from Epoch 1.
- **Loss Function**: `nn.CrossEntropyLoss(weight=cw_tensor)`.
- **Exact Class-Weight Configuration**: Inverse-frequency weights: $w_0 = 1.9018265$, $w_1 = 0.0981735$ (sum = 2.0).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `SwinUNETR` from scratch.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 1)**:
  - Macro-F1: **0.4863**
  - LGG Recall: **0.0000** (0/10)
  - Accuracy: **0.9465**
  - AUC: **0.7901**
  - Loss: **0.3401**
- **Locked-Test Prediction File**: `results/locked_test/protocol_B/swin_predictions.csv` (SHA-256: `c085604868854ebd69efe53706b4e6412086671e6ce10fb668e24d8af8e817ce`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9362**
  - Macro-F1: **0.4835**
  - LGG Recall: **0.0000** (0/12)
  - HGG Recall: **1.0000** (176/176)
  - AUC: **0.8643**
  - Precision: **0.9362**
  - Weighted-F1: **0.9055**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=0 & \text{FP}=12 \\ \text{FN}=0 & \text{TP}=176 \end{bmatrix}$

---

### 3.7 Configuration 7: Protocol C — ResNet18 MedicalNet
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_C/resnet18_medicalnet_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `resnet18_medicalnet_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `3d5b2989204f35f7c809628f71337dca629b1c1d83327e0d2e5c5768ed103160`
- **File Size**: 132,999,695 bytes
- **Epochs Trained**: 13
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 7**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 13, restoring weights from Epoch 7.
- **Loss Function**: `nn.CrossEntropyLoss(weight=cw_tensor)`.
- **Exact Class-Weight Configuration**: Minority $\times 1.25$ weights: $w_0 = 1.9206825$, $w_1 = 0.0793175$ (sum = 2.0).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `resnet18` initialized with MedicalNet 23-dataset weights.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 7)**:
  - Macro-F1: **0.7642**
  - LGG Recall: **0.7000** (7/10)
  - Accuracy: **0.9412**
  - AUC: **0.9110**
  - Loss: **0.2136**
- **Locked-Test Prediction File**: `results/locked_test/protocol_C/resnet18_predictions.csv` (SHA-256: `7151a184b30b59c1bc7d2879f5de64578207f80a48073b83a6efb6d5a7fcfdd9`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9255**
  - Macro-F1: **0.6884**
  - LGG Recall: **0.4167** (5/12)
  - HGG Recall: **0.9602** (169/176)
  - AUC: **0.9167**
  - Precision: **0.9602**
  - Weighted-F1: **0.9288**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=5 & \text{FP}=7 \\ \text{FN}=7 & \text{TP}=169 \end{bmatrix}$

---

### 3.8 Configuration 8: Protocol C — DenseNet121 3D
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_C/densenet121_3d_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `densenet121_3d_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `b519a8acac0f1d54b990a4d08e374e1cebda8470794d50fdb0c54f0089cf7ffa`
- **File Size**: 45,885,930 bytes
- **Epochs Trained**: 11
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 5**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 11, restoring weights from Epoch 5.
- **Loss Function**: `nn.CrossEntropyLoss(weight=cw_tensor)`.
- **Exact Class-Weight Configuration**: Minority $\times 1.25$ weights: $w_0 = 1.9206825$, $w_1 = 0.0793175$ (sum = 2.0).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `DenseNet121` from scratch.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 5)**:
  - Macro-F1: **0.7006**
  - LGG Recall: **0.6000** (6/10)
  - Accuracy: **0.9198**
  - AUC: **0.9181**
  - Loss: **0.2206**
- **Locked-Test Prediction File**: `results/locked_test/protocol_C/densenet121_predictions.csv` (SHA-256: `23ab3609521386179a694c0c7147e3c7074ecb69a625aae097527a19aed88bb1`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9202**
  - Macro-F1: **0.6527**
  - LGG Recall: **0.3333** (4/12)
  - HGG Recall: **0.9602** (169/176)
  - AUC: **0.9013**
  - Precision: **0.9548**
  - Weighted-F1: **0.9221**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=4 & \text{FP}=8 \\ \text{FN}=7 & \text{TP}=169 \end{bmatrix}$

---

### 3.9 Configuration 9: Protocol C — SwinUNETR 3D
- **Exact Checkpoint Path**: `results/protocol_archives/protocol_C/swin_transformer_3d_lgg_hgg_3d.pt`
- **Checkpoint Filename**: `swin_transformer_3d_lgg_hgg_3d.pt`
- **SHA-256 Hash**: `2346cda6d0563779279b8b53ba43586c3a2a6a392120a5cc84b8eb48e510a0ff`
- **File Size**: 70,432,451 bytes
- **Epochs Trained**: 14
- **Actual Epoch Whose Weights Were Saved/Restored**: **Epoch 8**
- **Checkpoint-Selection Criterion**: Validation Macro-F1 (primary, $\text{tolerance}=10^{-4}$) with Validation LGG Recall tie-breaker.
- **Early-Stopping Rule and Patience**: Patience = 6; triggers at Epoch 14, restoring weights from Epoch 8.
- **Loss Function**: `nn.CrossEntropyLoss(weight=cw_tensor)`.
- **Exact Class-Weight Configuration**: Minority $\times 1.25$ weights: $w_0 = 1.9206825$, $w_1 = 0.0793175$ (sum = 2.0).
- **Random Seed**: `42`.
- **Training Dataset / Split Source**: `train_manifest.json` ($N=876$).
- **Validation Dataset / Split Source**: `val_manifest.json` ($N=187$).
- **Locked Test Dataset Source**: `test_manifest.json` ($N=188$).
- **Preprocessing Configuration**: Standard 3D MONAI pipeline (RAS, 1.0mm iso, nonzero crop, $96^3$, nonzero z-score).
- **Augmentation Configuration**: Standard spatial/intensity training augmentations.
- **Model Initialization / Pretraining Source**: MONAI `SwinUNETR` from scratch.
- **Validation Metrics Associated with Checkpoint Selection (Epoch 8)**:
  - Macro-F1: **0.6831**
  - LGG Recall: **0.4000** (4/10)
  - Accuracy: **0.9358**
  - AUC: **0.9257**
  - Loss: **0.1931**
- **Locked-Test Prediction File**: `results/locked_test/protocol_C/swin_predictions.csv` (SHA-256: `773f6cc7f4f2f1a84e989391c5f279876c3a8a49b4edfa709e08083bbbda0b62`).
- **Final Reported Locked-Test Metrics**:
  - Accuracy: **0.9468**
  - Macro-F1: **0.6736**
  - LGG Recall: **0.2500** (3/12)
  - HGG Recall: **0.9943** (175/176)
  - AUC: **0.9735**
  - Precision: **0.9511**
  - Weighted-F1: **0.9431**
  - Confusion Matrix: $\begin{bmatrix} \text{TN}=3 & \text{FP}=9 \\ \text{FN}=1 & \text{TP}=175 \end{bmatrix}$

---

## 4. Critical Comparability Analysis (Protocol A vs B vs C)

### 4.1 Independent Variable Isolation
The intended independent variable across the three protocols is the class-weighting configuration supplied to `nn.CrossEntropyLoss`:
- **Protocol A**: $w_c = \text{None}$ (standard unweighted empirical loss).
- **Protocol B**: Inverse-frequency class weights $w_c = \frac{N}{K \cdot n_c}$, normalized such that $\sum_{c=1}^K w_c = K = 2$. Specifically: $w_0 = 1.9018$, $w_1 = 0.0982$.
- **Protocol C**: Inverse-frequency weights with minority-class multiplier $m = 1.25$, renormalized such that $\sum_{c=1}^K w_c = K = 2$. Specifically: $w_0 = 1.9207$, $w_1 = 0.0793$.

### 4.2 Normalization Control
A critical mathematical property of Protocols B and C is the **sum-to-$K$ normalization** ($w_{\text{final}} = w \cdot \frac{K}{\sum w}$). This ensures that the overall loss magnitude and expected gradient norm remain identical to unweighted Protocol A (where implicitly $w_0=1.0, w_1=1.0 \implies \sum w = 2.0$). Thus, effective learning rates were not artificially scaled between protocols.

### 4.3 Controlled Experimental Dimensions
The following factors were held strictly invariant across Protocols A, B, and C:
1. **Dataset Split Partitions**: All 9 experiments trained on identical 876 GLI subjects, validated on identical 187 GLI subjects, and evaluated on identical 188 locked-test subjects.
2. **Deterministic Seed**: `RANDOM_SEED = 42` applied before data loader creation and model weight initialization in all runs.
3. **Architectural Definitions**: Identical layer dimensions, kernel sizes, channels, pooling operations, and dropout rates ($p=0.4$) for ResNet18, DenseNet121, and SwinUNETR.
4. **Pretrained Weights**: The same `resnet_18_23dataset.pth` MedicalNet weights loaded for all three ResNet18 runs; the same seeded random initialization for DenseNet121 and SwinUNETR.
5. **Optimization**: `AdamW` optimizer with initial $\text{lr} = 10^{-4}$ and $\text{weight\_decay} = 10^{-4}$.
6. **LR Schedule**: `CosineAnnealingLR` with $T_{\max} = 30$ epochs, $\eta_{\min} = 10^{-6}$.
7. **Spatial Resolution**: $96 \times 96 \times 96$ voxels across all modalities.
8. **Batch Size & AMP**: Batch size of 2 with automated mixed precision (fp16) on CUDA.
9. **Decision Threshold**: Probability threshold $\tau = 0.5$ for binary classification ($\hat{y} = 1 \iff P(\text{HGG}) \ge 0.5$).

### 4.4 Identification of Consequential Dynamic Variables
The audit identified one variable that differed across runs: **Total Epochs Trained** (ranging from 7 to 25 epochs) and **Selected Best Epoch** (ranging from Epoch 1 to Epoch 19).
- **Nature of Difference**: This variation is not an uncontrolled experimental confounding factor, but rather the deterministic consequence of applying the exact same stopping criterion (`EarlyStopping(patience=6)`) to distinct loss functions. Different class-weight penalties naturally produce different loss surfaces, leading models to plateau at different points during training.

---

## 5. Critical Provenance Question: Procedure-Designated vs Retrospective Cherry-Picking

### 5.1 Formulation of the Provenance Question
> *Determine whether the currently archived Protocol A/B/C checkpoints are genuinely the checkpoints designated by the project's documented training/selection procedure, rather than simply the highest-performing checkpoints observed retrospectively.*

### 5.2 Forensic Verification Steps
To resolve this question definitively, five independent forensic checks were conducted:

1. **Step-by-Step History Reconstruction**: The `history` arrays in all 9 `.pt` checkpoints and history JSON files were parsed. The intra-run metric sequence was simulated using the documented early-stopping rule:
   $$\text{Improved} \iff \text{Macro-F1} > \text{Best} + 10^{-4} \quad \lor \quad (|\text{Macro-F1} - \text{Best}| \le 10^{-4} \land \text{LGG Recall} > \text{Best LGG})$$
   In all 9 cases, the simulated best epoch exactly matched the recorded `saved_weights_epoch` and the stored metadata `best_val_macro_f1` and `best_val_lgg_recall`.
2. **Intra-Run Weight Buffer Capture**: The training code calls `stopper.restore(model)` immediately upon loop exit, ensuring that the in-memory weights correspond to the best validation epoch rather than the final training epoch.
3. **Execution Timeline & Log Cross-Referencing**: Training logs record the exact epoch timestamps and log lines confirming that checkpoints were saved to `models/` at the conclusion of training.
4. **Separation of Locked-Test Evaluation**: Locked-test evaluation scripts (`evaluate_locked_test.py` and `run_day5_phase1_locked_test.py`) loaded the pre-existing archived checkpoints strictly in evaluation mode (`torch.no_grad()`). No parameter grid search or retrospective checkpoint swapping occurred.
5. **Validation vs Locked-Test Metric Differences**: For models like SwinUNETR Protocol A and B, the model achieved validation Macro-F1 = 0.4863 (LGG Recall = 0.0) at Epoch 1 and stopped at Epoch 7. If retrospective cherry-picking had occurred, alternative epochs or models would have been substituted. The exact preservation of this failure case confirms fidelity to the documented procedure.

### 5.3 Provenance Verdict
**UNAMBIGUOUS RESOLUTION: GENUINELY DESIGNATED BY PROCEDURE**  
The archived checkpoints are conclusively verified to be the genuine checkpoints produced by the documented training and selection procedure.

---

## 6. Pre- and Post-Audit Integrity Verification

All SHA-256 hashes were computed prior to audit execution and re-verified upon completion:

| Subdirectory / Component | Total Files | Modifications Expected | Modifications Observed | Status |
|---|---|---|---|---|
| `results/protocol_archives/protocol_A/` | 9 | 0 | 0 | **VERIFIED UNCHANGED** |
| `results/protocol_archives/protocol_B/` | 7 | 0 | 0 | **VERIFIED UNCHANGED** |
| `results/protocol_archives/protocol_C/` | 9 | 0 | 0 | **VERIFIED UNCHANGED** |
| `results/locked_test/` | 39 | 0 | 0 | **VERIFIED UNCHANGED** |

---

## 7. Master Comparison & Summary Tables

### 7.1 Cross-Protocol Comparison Table (Validation vs Locked-Test)

| Protocol | Model | Trained Ep | Restored Ep | Val Macro-F1 | Val LGG Recall | Val Acc | Test Acc | Test Macro-F1 | Test LGG Recall | Test HGG Recall | Test AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | ResNet18 MedNet | 25 | **19** | 0.7642 | 0.7000 | 0.9412 | **0.9734** | **0.8842** | **0.7500** | 0.9886 | **0.9830** |
| **A** | DenseNet121 3D | 11 | **5** | 0.6785 | 0.5000 | 0.9198 | 0.9468 | 0.7934 | 0.6667 | 0.9659 | 0.9422 |
| **A** | SwinUNETR 3D | 7 | **1** | 0.4863 | 0.0000 | 0.9465 | 0.9362 | 0.4835 | 0.0000 | 1.0000 | 0.9337 |
| **B** | ResNet18 MedNet | 13 | **7** | 0.7101 | 0.7000 | 0.9144 | 0.9149 | 0.7267 | 0.6667 | 0.9318 | 0.8939 |
| **B** | DenseNet121 3D | 11 | **5** | 0.6785 | 0.5000 | 0.9198 | 0.9149 | 0.7102 | 0.5833 | 0.9375 | 0.9460 |
| **B** | SwinUNETR 3D | 7 | **1** | 0.4863 | 0.0000 | 0.9465 | 0.9362 | 0.4835 | 0.0000 | 1.0000 | 0.8643 |
| **C** | ResNet18 MedNet | 13 | **7** | 0.7642 | 0.7000 | 0.9412 | 0.9255 | 0.6884 | 0.4167 | 0.9602 | 0.9167 |
| **C** | DenseNet121 3D | 11 | **5** | 0.7006 | 0.6000 | 0.9198 | 0.9202 | 0.6527 | 0.3333 | 0.9602 | 0.9013 |
| **C** | SwinUNETR 3D | 14 | **8** | 0.6831 | 0.4000 | 0.9358 | 0.9468 | 0.6736 | 0.2500 | 0.9943 | 0.9735 |

---

## 8. Audit Conclusion & Output Manifest

All required artifacts have been generated in `results/reliability_analysis/day7_provenance_audit/`:
1. `protocol_provenance_master.csv` — Comprehensive 39-column master dataset.
2. `protocol_provenance_audit.md` — Full markdown forensic report.
3. `protocol_comparability_matrix.csv` — Feature-by-feature comparability matrix across protocols.
4. `checkpoint_provenance.csv` — Detailed checkpoint provenance and selection table.
5. `day7_audit_manifest.json` — Complete JSON manifest with cryptographic SHA-256 signatures.
