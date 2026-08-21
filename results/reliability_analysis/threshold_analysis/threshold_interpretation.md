# Decision-Threshold Sensitivity: Clinical & Reliability Interpretation

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Scope**: 9 Canonical Model Checkpoints across Protocols A, B, C  
**Date**: 2026-08-17  
**Audit Mode**: STRICT READ-ONLY FORENSIC INTERPRETATION  

---

## 1. Executive Interpretation of Findings

### 1. The "0 LGG Recall" Phenomenon: Representation vs. Calibration
A pivotal discovery of this analysis is that **poor LGG recall at threshold 0.50 is largely an operating-point / thresholding problem, rather than a total failure of latent representation learning**.

* **The SwinUNETR Protocol A Case**:
  - At default $\tau = 0.50$, SwinUNETR Protocol A classified 0 of 12 test LGG cases (LGG Recall = 0.0000, Macro-F1 = 0.4835).
  - Yet, its ROC-AUC was **0.9337** on the test set and **0.8780** on validation.
  - When the decision threshold $\tau_{\text{LGG}}$ was adjusted to $0.03$ (selected on validation data), test LGG Recall jumped from **0.0000 to 1.0000 (12/12 LGG cases detected)**, raising Macro-F1 to **0.6234**.
  - **Scientific Meaning**: The model learned high-dimensional features capable of ranking LGG volumes distinctly from HGG volumes, but its output probabilities were heavily compressed near zero (typically $P(\text{LGG}) \in [0.03, 0.20]$) due to unweighted cross-entropy loss under 95:5 imbalance.

* **The SwinUNETR Protocol C Case**:
  - At default $\tau = 0.50$: LGG Recall = 0.2500, Macro-F1 = 0.6736, AUC = 0.9735.
  - Under Rule B ($\tau = 0.24$ selected on validation): LGG Recall jumped to **0.6667 (8/12)**, Specificity remained **0.9659 (170/176)**, and Macro-F1 rose to **0.7934**.
  - Under Rule C ($\tau = 0.18$ selected on validation): LGG Recall jumped to **0.8333 (10/12)**, Specificity remained **0.9545 (168/176)**, and Macro-F1 rose to **0.8189**.

---

## 2. Default (0.50) vs. Development-Selected Threshold Comparison

| Model | Protocol | Test ROC-AUC | Default Threshold ($\tau=0.50$) | | | Rule B Threshold (Spec $\ge 90\%$) | | | Rule C Threshold (Max Bal Acc) | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | | | **LGG Rec** | **HGG Spec** | **Macro-F1** | **$\tau^*$** | **LGG Rec** | **HGG Spec** | **$\tau^*$** | **LGG Rec** | **HGG Spec** |
| **ResNet18 MedicalNet** | **A** | **0.9830** | **0.7500** | **0.9886** | **0.8842** | 0.23 | **0.8333** | 0.9489 | 0.22 | **0.9167** | 0.9432 |
| **DenseNet121 3D** | **A** | **0.9422** | 0.6667 | 0.9659 | 0.7934 | 0.27 | 0.6667 | 0.9432 | 0.05 | 1.0000 | 0.7443 |
| **SwinUNETR 3D** | **A** | **0.9337** | 0.0000 | 1.0000 | 0.4835 | 0.04 | 0.0000 | 1.0000 | 0.03 | 1.0000 | 0.7727 |
| **ResNet18 MedicalNet** | **B** | **0.8939** | 0.6667 | 0.9318 | 0.7267 | 0.62 | 0.6667 | 0.9716 | 0.62 | 0.6667 | 0.9716 |
| **DenseNet121 3D** | **B** | **0.9460** | 0.5833 | 0.9375 | 0.7102 | 0.40 | **0.7500** | 0.9148 | 0.11 | 1.0000 | 0.7557 |
| **SwinUNETR 3D** | **B** | **0.8643** | 0.0000 | 1.0000 | 0.4835 | 0.06 | 0.2500 | 0.9943 | 0.06 | 0.2500 | 0.9943 |
| **ResNet18 MedicalNet** | **C** | **0.9167** | 0.4167 | 0.9602 | 0.6884 | 0.52 | 0.4167 | 0.9602 | 0.08 | **0.8333** | 0.8068 |
| **DenseNet121 3D** | **C** | **0.9013** | 0.3333 | 0.9602 | 0.6527 | 0.49 | 0.3333 | 0.9602 | 0.06 | **0.9167** | 0.7159 |
| **SwinUNETR 3D** | **C** | **0.9735** | 0.2500 | 0.9943 | 0.6736 | 0.24 | **0.6667** | 0.9659 | 0.18 | **0.8333** | 0.9545 |

*Note: ROC-AUC is completely invariant to decision threshold.*

---

## 3. Threshold Tuning vs. Probability Calibration

In medical AI reporting, it is essential to maintain precise terminology:

1. **Threshold Tuning (Operating-Point Selection)**:
   - Modifies the binary classification boundary $\tau$ applied to existing output probabilities.
   - Shifts the operating point along the existing ROC curve.
   - Alters Accuracy, Macro-F1, Sensitivity, Specificity, and Confusion Matrix.
   - **Does NOT alter ROC-AUC, ECE (Expected Calibration Error), or Brier Score.**

2. **Probability Calibration (e.g., Temperature Scaling, Isotonic Regression)**:
   - Transforms the probability distribution $P(y=k \mid x)$ to reflect true empirical posterior probabilities.
   - Directly improves ECE and Brier Score.
   - Threshold tuning without probability recalibration adjusts decision boundaries but leaves uncalibrated probabilities uncalibrated.

---

## 4. Is Threshold Tuning Sufficient to Solve the Class Imbalance Problem?

### Answer: **PARTIALLY, BUT NOT COMPLETELY.**

1. **Where Threshold Tuning Succeeds**:
   - For models with **high baseline discriminability** ($\text{AUC} > 0.95$):
     - **ResNet18 Protocol A** ($\text{AUC} = 0.9830$): Moving threshold to $\tau=0.22$ increases LGG Recall from 75.0% to **91.7%** (11/12 LGG detected) with only 10 false alarms across 176 HGG patients (Specificity = 94.3%).
     - **SwinUNETR Protocol C** ($\text{AUC} = 0.9735$): Moving threshold to $\tau=0.18$ increases LGG Recall from 25.0% to **83.3%** (10/12 LGG detected) with Specificity = 95.5%.

2. **Where Threshold Tuning Fails**:
   - For models with **lower discriminability** (e.g., SwinUNETR Protocol B, $\text{AUC} = 0.8643$):
     - Forcing high LGG recall causes severe specificity collapse ($>30-40$ false positives on HGG).
   - In unweighted models with extreme probability compression (SwinUNETR Protocol A):
     - The optimal threshold ($\tau = 0.03$) is located in an unstable, noise-sensitive probability region.

3. **Conclusion for Training-Level Interventions (e.g., Focal Loss)**:
   - Threshold tuning is a post-hoc operating-point shift, not an optimization of representation geometry.
   - Training-level objectives (such as **Focal Loss** or balanced margin losses) dynamically penalize easy negative HGG samples during backpropagation, producing better-calibrated output representations and wider decision margins.
   - **Focal Loss therefore remains scientifically justified** for future research.
