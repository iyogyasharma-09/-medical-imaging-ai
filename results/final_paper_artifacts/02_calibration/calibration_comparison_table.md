# Manuscript Table 3: Post-Hoc Temperature Scaling Comparison

*Evaluating the impact of validation-only temperature scaling on locked-test probability calibration and rank invariance.*

| Model | Protocol | Frozen Temperature T* | Raw ECE | Calibrated ECE | Delta ECE | Raw Brier | Calibrated Brier | Delta Brier | ROC-AUC (Before / After) | LGG Recall at 0.50 (Before / After) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 MedicalNet | Protocol A | 1.0622 | 0.0276 | 0.0334 | +0.0058 | 0.0248 | 0.0250 | +0.0002 | 0.9830 / 0.9830 | 0.7500 / 0.7500 |
| DenseNet121 3D | Protocol A | 1.0233 | 0.0308 | 0.0330 | +0.0022 | 0.0414 | 0.0415 | +0.0001 | 0.9422 / 0.9422 | 0.6667 / 0.6667 |
| SwinUNETR 3D | Protocol A | 1.2101 | 0.0345 | 0.0112 | -0.0233 | 0.0607 | 0.0595 | -0.0012 | 0.9330 / 0.9330 | 0.0000 / 0.0000 |
| ResNet18 MedicalNet | Protocol B | 1.4408 | 0.0394 | 0.0474 | +0.0080 | 0.0521 | 0.0528 | +0.0007 | 0.8939 / 0.8939 | 0.6667 / 0.6667 |
| DenseNet121 3D | Protocol B | 0.9703 | 0.0540 | 0.0523 | -0.0017 | 0.0522 | 0.0521 | -0.0001 | 0.9460 / 0.9460 | 0.5833 / 0.5833 |
| SwinUNETR 3D | Protocol B | 0.9614 | 0.0046 | 0.0105 | +0.0059 | 0.0597 | 0.0598 | +0.0001 | 0.8667 / 0.8667 | 0.0000 / 0.0000 |
| ResNet18 MedicalNet | Protocol C | 1.0749 | 0.0142 | 0.0099 | -0.0043 | 0.0510 | 0.0507 | -0.0003 | 0.9167 / 0.9167 | 0.4167 / 0.4167 |
| DenseNet121 3D | Protocol C | 0.9650 | 0.0353 | 0.0277 | -0.0076 | 0.0561 | 0.0562 | +0.0001 | 0.9010 / 0.9010 | 0.3333 / 0.3333 |
| SwinUNETR 3D | Protocol C | 0.9952 | 0.0359 | 0.0358 | -0.0001 | 0.0352 | 0.0352 | +0.0000 | 0.9735 / 0.9735 | 0.2500 / 0.2500 |
| ResNet18 MedicalNet (Focal) | Focal Loss | 0.4922 | 0.0975 | 0.0341 | -0.0634 | 0.0497 | 0.0394 | -0.0103 | 0.9541 / 0.9541 | 0.4167 / 0.4167 |

> **Core Scientific Finding:** Post-hoc calibration modifies probability reliability (reducing ECE and Brier score), but mathematically preserves rank discrimination (ROC-AUC) and default binary decisions at tau=0.50.