# Core Paper Finding #2: High ROC-AUC Does Not Guarantee Clinical Calibration or Reliability

## 1. Primary Research Statement
> **"Threshold-agnostic discrimination (ROC-AUC) is uncoupled from probability calibration (ECE) and minority-class sensitivity; high AUC does not imply clinical reliability."**

---

## 2. Empirical Support

Case study comparison between **SwinUNETR 3D (Protocol C)** and **ResNet18 MedicalNet (Protocol A)**:

- **SwinUNETR 3D (Protocol C)**:
  - **AUC**: **0.9735** (Exceedingly high discrimination ranking)
  - **LGG Recall**: **0.2500** (Fails to detect 9 out of 12 LGG subjects)
  - **High-Confidence Errors**: 4 misclassifications committed with high probability.
  - **Clinical Safety Gate**: `NON-VIABLE (Gated)`

- **ResNet18 MedicalNet (Protocol A)**:
  - **AUC**: **0.9830**
  - **LGG Recall**: **0.7500** (Detects 9 out of 12 LGG subjects)
  - **ECE**: **0.0276** (Near-perfect probability calibration)
  - **High-Confidence Errors**: Only 1 error $\ge 0.80$.
  - **Clinical Safety Gate**: `VIABLE`

---

## 3. Methodological Implication for Medical AI
Relying solely on ROC-AUC for 3D medical image grading is dangerous. A model can rank probability outputs effectively across arbitrary cutoffs while failing completely at the operational decision boundary (0.5) and outputting miscalibrated confidence values.
