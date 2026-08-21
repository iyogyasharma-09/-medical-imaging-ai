# Dedicated Focal Loss Evaluation & Operating-Point Table

*Comparison of Cross-Entropy Baseline vs Focal Loss across Raw, Calibrated, and Threshold-Adjusted conditions on the locked test cohort.*

| Configuration | Loss | Threshold | Calibrated | Accuracy | Macro-F1 | ROC-AUC | LGG Recall | HGG Recall | ECE | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 Protocol A (CE Baseline) [Raw] | Standard Cross-Entropy | 0.5 | No | 0.9734 | 0.8842 | 0.983 | 0.75 | 0.9886 | 0.0276 | 0.0248 |
| ResNet18 Focal Loss [Raw] | Focal Loss (alpha=1.0, gamma=2.0) | 0.5 | No | 0.9362 | 0.7103 | 0.9541 | 0.4167 | 0.9716 | 0.0975 | 0.0497 |
| ResNet18 Focal Loss [Calibrated T*=0.4923] | Focal Loss (alpha=1.0, gamma=2.0) | 0.5 | Yes (T*=0.4923) | 0.9362 | 0.7103 | 0.9541 | 0.4167 | 0.9716 | 0.0341 | 0.0394 |
| ResNet18 Focal Loss [Calibrated + Max F1 Operating Point] | Focal Loss (alpha=1.0, gamma=2.0) | 0.35 | Yes (T*=0.4923) | 0.9362 | 0.7685 | 0.9541 | 0.6667 | 0.9545 | 0.0341 | 0.0394 |
| ResNet18 Focal Loss [Calibrated + High Sensitivity Operating Point] | Focal Loss (alpha=1.0, gamma=2.0) | 0.2 | Yes (T*=0.4923) | 0.883 | 0.7052 | 0.9541 | 0.8333 | 0.8864 | 0.0341 | 0.0394 |
| ResNet18 Focal Loss [Calibrated + Max BalAcc Operating Point] | Focal Loss (alpha=1.0, gamma=2.0) | 0.1 | Yes (T*=0.4923) | 0.8457 | 0.6815 | 0.9541 | 1.0 | 0.8352 | 0.0341 | 0.0394 |

> **Core Scientific Interpretation:** Focal Loss did not eliminate the class-specific operating limitation at default threshold (tau=0.50), despite retaining substantial discriminatory ability (AUC=0.9541). Post-hoc temperature scaling restored probability calibration (ECE: 9.75% -> 3.41%), enabling principled selection of high-sensitivity operating points without altering the underlying ranking.