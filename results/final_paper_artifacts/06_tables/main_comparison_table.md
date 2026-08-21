# Manuscript Table 1: Main Baseline & Intervention Comparison

*Locked test cohort (n=188: 12 LGG, 176 HGG). Demonstrating multi-dimensional evaluation across architectures and loss protocols.*

| Model | Protocol | Loss Type | ROC-AUC | LGG Recall (tau=0.50) | HGG Recall (tau=0.50) | Macro-F1 (tau=0.50) | Raw ECE | Calibrated ECE | Calibrated Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 MedicalNet | Protocol A | Standard Unweighted CE | 0.9830 | 0.7500 | 0.9886 | 0.8842 | 0.0276 | 0.0334 | 0.0250 |
| DenseNet121 3D | Protocol A | Standard Unweighted CE | 0.9422 | 0.6667 | 0.9659 | 0.7934 | 0.0308 | 0.0330 | 0.0415 |
| SwinUNETR 3D | Protocol A | Standard Unweighted CE | 0.9330 | 0.0000 | 1.0000 | 0.4835 | 0.0345 | 0.0112 | 0.0595 |
| ResNet18 MedicalNet | Protocol B | Inverse-Frequency CE | 0.8939 | 0.6667 | 0.9318 | 0.7267 | 0.0394 | 0.0474 | 0.0528 |
| DenseNet121 3D | Protocol B | Inverse-Frequency CE | 0.9460 | 0.5833 | 0.9375 | 0.7102 | 0.0540 | 0.0523 | 0.0521 |
| SwinUNETR 3D | Protocol B | Inverse-Frequency CE | 0.8667 | 0.0000 | 1.0000 | 0.4835 | 0.0046 | 0.0105 | 0.0598 |
| ResNet18 MedicalNet | Protocol C | Adjusted Minority CE | 0.9167 | 0.4167 | 0.9602 | 0.6884 | 0.0142 | 0.0099 | 0.0507 |
| DenseNet121 3D | Protocol C | Adjusted Minority CE | 0.9010 | 0.3333 | 0.9602 | 0.6527 | 0.0353 | 0.0277 | 0.0562 |
| SwinUNETR 3D | Protocol C | Adjusted Minority CE | 0.9735 | 0.2500 | 0.9943 | 0.6736 | 0.0359 | 0.0358 | 0.0352 |
| ResNet18 MedicalNet (Focal) | Focal Loss | Focal Loss (alpha=1.0, gamma=2.0) | 0.9541 | 0.4167 | 0.9716 | 0.7103 | 0.0975 | 0.0341 | 0.0394 |