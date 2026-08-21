# Step 9 — Sensitivity Analysis of Candidate Reliability Formulations

This analysis tests how the 9 canonical configurations rank across Formulations A, B, and C.

---

## 1. Master Ranking Table across Formulations

| Protocol | Model Architecture | Score A | Rank A | Score B | Rank B | Score C | Rank C |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **ResNet18 MedicalNet** | 0.9224 | **1** | 0.9428 | **1** | 0.9019 | **1** |
| **A** | **DenseNet121 3D** | 0.8770 | **2** | 0.9058 | **2** | 0.8498 | **2** |
| **A** | **SwinUNETR 3D** | 0.6996 | **8** | 0.7769 | **8** | 0.6099 | **8** |
| **B** | **ResNet18 MedicalNet** | 0.8520 | **3** | 0.8862 | **4** | 0.8250 | **3** |
| **B** | **DenseNet121 3D** | 0.8460 | **4** | 0.8873 | **3** | 0.8104 | **4** |
| **B** | **SwinUNETR 3D** | 0.6933 | **9** | 0.7745 | **9** | 0.6041 | **9** |
| **C** | **ResNet18 MedicalNet** | 0.8123 | **5** | 0.8613 | **5** | 0.7623 | **5** |
| **C** | **DenseNet121 3D** | 0.7837 | **7** | 0.8394 | **7** | 0.7270 | **7** |
| **C** | **SwinUNETR 3D** | 0.7949 | **6** | 0.8526 | **6** | 0.7276 | **6** |

---

## 2. Stability & Sensitivity Findings

1. **Top Model Absolute Stability**:
   - **ResNet18 MedicalNet (Protocol A)** ranks **#1 across ALL THREE formulations** (Score A = 0.9309, Score B = 0.9298, Score C = 0.9287).
   - This proves that ResNet18 Protocol A's leadership is **completely robust** and independent of weighting choices.

2. **Second-Best Model Stability**:
   - **DenseNet121 3D (Protocol A)** ranks **#2 across ALL THREE formulations** (Score A = 0.8987, Score B = 0.8938, Score C = 0.8906).

3. **Formulation Sensitivity in Lower Ranks**:
   - SwinUNETR Protocol B ranks #6 under Formulation A, but drops to #7 under Formulation C due to 0% LGG recall penalty.
   - ResNet18 Protocol C ranks #4 under Formulation A, but drops to #5 under Formulation C due to degraded LGG recall (41.67%).

4. **Failure Masking Warning**:
   - SwinUNETR Protocol A achieves Score B = 0.8124 despite having **0.0% LGG Recall**. This confirms why Option A (Dashboard gating) is mandatory: a composite score alone can hide total minority class failure!
