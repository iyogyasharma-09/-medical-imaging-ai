# Manuscript Table 4: Descriptive Calibrated Operating Points

*Illustrative operating-point trade-offs across calibrated decision thresholds on the locked test cohort.*

| Model Configuration | Calibrated Threshold tau_LGG | Corresponding theta_HGG | LGG Recall (Sensitivity) | HGG Recall (Specificity) | Accuracy | Macro-F1 | Balanced Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 (Focal Loss) | 0.50 | 0.50 | 0.4167 (5/12) | 0.9716 (171/176) | 0.9362 | 0.7103 | 0.6941 |
| ResNet18 (Focal Loss) | 0.40 | 0.60 | 0.5833 (7/12) | 0.9602 (169/176) | 0.9362 | 0.7521 | 0.7718 |
| ResNet18 (Focal Loss) | 0.35 | 0.65 | 0.6667 (8/12) | 0.9545 (168/176) | 0.9362 | 0.7685 | 0.8106 |
| ResNet18 (Focal Loss) | 0.30 | 0.70 | 0.6667 (8/12) | 0.9489 (167/176) | 0.9309 | 0.7571 | 0.8078 |
| ResNet18 (Focal Loss) | 0.25 | 0.75 | 0.7500 (9/12) | 0.9261 (163/176) | 0.9149 | 0.7413 | 0.8381 |
| ResNet18 (Focal Loss) | 0.20 | 0.80 | 0.8333 (10/12) | 0.8864 (156/176) | 0.8830 | 0.7052 | 0.8598 |
| ResNet18 (Focal Loss) | 0.15 | 0.85 | 0.9167 (11/12) | 0.8807 (155/176) | 0.8830 | 0.7169 | 0.8987 |
| ResNet18 (Focal Loss) | 0.10 | 0.90 | 1.0000 (12/12) | 0.8352 (147/176) | 0.8457 | 0.6815 | 0.9176 |
| ResNet18 (Protocol A - CE) | 0.50 | 0.50 | 0.7500 (9/12) | 0.9886 (174/176) | 0.9734 | 0.8842 | 0.8693 |
| ResNet18 (Protocol A - CE) | 0.40 | 0.60 | 0.7500 (9/12) | 0.9716 (171/176) | 0.9574 | 0.8347 | 0.8608 |
| ResNet18 (Protocol A - CE) | 0.35 | 0.65 | 0.7500 (9/12) | 0.9716 (171/176) | 0.9574 | 0.8347 | 0.8608 |
| ResNet18 (Protocol A - CE) | 0.30 | 0.70 | 0.7500 (9/12) | 0.9489 (167/176) | 0.9362 | 0.7827 | 0.8494 |
| ResNet18 (Protocol A - CE) | 0.25 | 0.75 | 0.8333 (10/12) | 0.9489 (167/176) | 0.9415 | 0.8066 | 0.8911 |
| ResNet18 (Protocol A - CE) | 0.20 | 0.80 | 0.9167 (11/12) | 0.9318 (164/176) | 0.9309 | 0.7952 | 0.9242 |
| ResNet18 (Protocol A - CE) | 0.15 | 0.85 | 1.0000 (12/12) | 0.9261 (163/176) | 0.9309 | 0.8052 | 0.9631 |
| ResNet18 (Protocol A - CE) | 0.10 | 0.90 | 1.0000 (12/12) | 0.8920 (157/176) | 0.8989 | 0.7505 | 0.9460 |
| DenseNet121 (Protocol A - CE) | 0.50 | 0.50 | 0.6667 (8/12) | 0.9659 (170/176) | 0.9468 | 0.7934 | 0.8163 |
| DenseNet121 (Protocol A - CE) | 0.40 | 0.60 | 0.6667 (8/12) | 0.9545 (168/176) | 0.9362 | 0.7685 | 0.8106 |
| DenseNet121 (Protocol A - CE) | 0.35 | 0.65 | 0.6667 (8/12) | 0.9489 (167/176) | 0.9309 | 0.7571 | 0.8078 |
| DenseNet121 (Protocol A - CE) | 0.30 | 0.70 | 0.6667 (8/12) | 0.9432 (166/176) | 0.9255 | 0.7464 | 0.8049 |
| DenseNet121 (Protocol A - CE) | 0.25 | 0.75 | 0.6667 (8/12) | 0.9318 (164/176) | 0.9149 | 0.7267 | 0.7992 |
| DenseNet121 (Protocol A - CE) | 0.20 | 0.80 | 0.6667 (8/12) | 0.8920 (157/176) | 0.8777 | 0.6710 | 0.7794 |
| DenseNet121 (Protocol A - CE) | 0.15 | 0.85 | 0.7500 (9/12) | 0.8636 (152/176) | 0.8564 | 0.6592 | 0.8068 |
| DenseNet121 (Protocol A - CE) | 0.10 | 0.90 | 0.9167 (11/12) | 0.8409 (148/176) | 0.8457 | 0.6711 | 0.8788 |
| SwinUNETR (Protocol C - Adj-Min) | 0.50 | 0.50 | 0.2500 (3/12) | 0.9943 (175/176) | 0.9468 | 0.6736 | 0.6222 |
| SwinUNETR (Protocol C - Adj-Min) | 0.40 | 0.60 | 0.5000 (6/12) | 0.9886 (174/176) | 0.9574 | 0.7888 | 0.7443 |
| SwinUNETR (Protocol C - Adj-Min) | 0.35 | 0.65 | 0.5000 (6/12) | 0.9716 (171/176) | 0.9415 | 0.7453 | 0.7358 |
| SwinUNETR (Protocol C - Adj-Min) | 0.30 | 0.70 | 0.5833 (7/12) | 0.9716 (171/176) | 0.9468 | 0.7775 | 0.7775 |
| SwinUNETR (Protocol C - Adj-Min) | 0.25 | 0.75 | 0.6667 (8/12) | 0.9659 (170/176) | 0.9468 | 0.7934 | 0.8163 |
| SwinUNETR (Protocol C - Adj-Min) | 0.20 | 0.80 | 0.6667 (8/12) | 0.9602 (169/176) | 0.9415 | 0.7805 | 0.8134 |
| SwinUNETR (Protocol C - Adj-Min) | 0.15 | 0.85 | 0.8333 (10/12) | 0.9375 (165/176) | 0.9309 | 0.7841 | 0.8854 |
| SwinUNETR (Protocol C - Adj-Min) | 0.10 | 0.90 | 1.0000 (12/12) | 0.9148 (161/176) | 0.9202 | 0.7854 | 0.9574 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.50 | 0.50 | 0.5833 (7/12) | 0.9375 (165/176) | 0.9149 | 0.7102 | 0.7604 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.40 | 0.60 | 0.7500 (9/12) | 0.9148 (161/176) | 0.9043 | 0.7235 | 0.8324 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.35 | 0.65 | 0.8333 (10/12) | 0.9091 (160/176) | 0.9043 | 0.7365 | 0.8712 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.30 | 0.70 | 0.8333 (10/12) | 0.8977 (158/176) | 0.8936 | 0.7202 | 0.8655 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.25 | 0.75 | 0.9167 (11/12) | 0.8693 (153/176) | 0.8723 | 0.7028 | 0.8930 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.20 | 0.80 | 1.0000 (12/12) | 0.8523 (150/176) | 0.8617 | 0.7001 | 0.9261 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.15 | 0.85 | 1.0000 (12/12) | 0.7898 (139/176) | 0.8032 | 0.6380 | 0.8949 |
| DenseNet121 (Protocol B - Inv-Freq) | 0.10 | 0.90 | 1.0000 (12/12) | 0.7443 (131/176) | 0.7606 | 0.6006 | 0.8722 |

> **Important Note:** These results represent descriptive operating-point sweeps on the locked test cohort to demonstrate the continuous sensitivity/specificity trade-off. Prospective clinical deployment requires pre-specifying thresholds on independent validation data under an explicit clinical utility function.