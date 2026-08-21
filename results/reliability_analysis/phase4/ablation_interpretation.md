# Phase 4 — Ablation Study #1: Research Synthesis & Interpretation

## Executive Summary

Ablation Study #1 evaluates the impact of class-weighting optimization strategies (Protocols A, B, C) across three 3D neural architectures evaluated on the locked BraTS 2023 test set.

---

## 1. Master Pairwise Transition Deltas

| Architecture | Transition | $\Delta$ Macro-F1 | Macro-F1 Effect | $\Delta$ LGG Recall | LGG Recall Effect | $\Delta$ AUC | $\Delta$ ECE | $\Delta$ Brier | $\Delta$ HighConfErrRate | $\Delta R_B$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18 MedicalNet** | **A -> B** | -0.1575 | Deterioration | **-0.0833** | **Deterioration** | -0.0891 | +0.0118 | +0.0546 | +0.0266 | -0.0566 |
| **ResNet18 MedicalNet** | **B -> C** | -0.0383 | Deterioration | **-0.2500** | **Deterioration** | +0.0228 | -0.0252 | -0.0023 | +0.0000 | -0.0249 |
| **ResNet18 MedicalNet** | **A -> C** | -0.1958 | Deterioration | **-0.3333** | **Deterioration** | -0.0663 | -0.0134 | +0.0523 | +0.0266 | -0.0815 |
| **DenseNet121 3D** | **A -> B** | -0.0832 | Deterioration | **-0.0834** | **Deterioration** | +0.0038 | +0.0232 | +0.0216 | -0.0213 | -0.0185 |
| **DenseNet121 3D** | **B -> C** | -0.0575 | Deterioration | **-0.2500** | **Deterioration** | -0.0447 | -0.0187 | +0.0078 | +0.0319 | -0.0479 |
| **DenseNet121 3D** | **A -> C** | -0.1407 | Deterioration | **-0.3334** | **Deterioration** | -0.0409 | +0.0045 | +0.0294 | +0.0106 | -0.0664 |
| **SwinUNETR 3D** | **A -> B** | +0.0000 | Negligible | **+0.0000** | **Negligible** | -0.0694 | -0.0299 | -0.0020 | +0.0000 | -0.0024 |
| **SwinUNETR 3D** | **B -> C** | +0.1901 | Improvement | **+0.2500** | **Improvement** | +0.1092 | +0.0313 | -0.0490 | -0.0425 | +0.0781 |
| **SwinUNETR 3D** | **A -> C** | +0.1901 | Improvement | **+0.2500** | **Improvement** | +0.0398 | +0.0014 | -0.0510 | -0.0425 | +0.0757 |

---

## 2. Hypothesis Testing & Evaluation of Research Statements

### Statement 1: *"Class weighting does not universally improve minority-class classification; its effect is architecture-dependent."*
- **Status**: **FULLY SUPPORTED BY EMPIRICAL EVIDENCE**.
- **Evidence**:
  - For **ResNet18 MedicalNet** (pretrained 3D CNN), adding class weights caused severe LGG Recall deterioration ($0.7500 \rightarrow 0.6667 \rightarrow 0.4167$) and Macro-F1 drop ($0.8842 \rightarrow 0.7267 \rightarrow 0.6884$).
  - For **DenseNet121 3D** (dense feature reuse CNN), adding class weights similarly degraded LGG Recall ($0.6667 \rightarrow 0.5833 \rightarrow 0.3333$) and Macro-F1 ($0.7934 \rightarrow 0.7102 \rightarrow 0.6527$).
  - For **SwinUNETR 3D** (randomly initialized vision transformer), unweighted training completely collapsed onto the majority class (0.0% LGG Recall in A and B), whereas Protocol C provided the necessary gradient signal to achieve non-zero minority detection (25.0% LGG Recall, +19.01% Macro-F1).

### Statement 2: *"Adding class weights always improves minority classification."*
- **Status**: **UNSUPPORTED AND REFUTED BY EXPERIMENTAL RESULTS**.
- **Evidence**: In 2 out of 3 architectures (ResNet18 and DenseNet121), adding class weights **reduced** minority LGG Recall by 33.3% and degraded overall Macro-F1 by 14.1–19.6%.

---

## 3. Detailed Answers to Core Ablation Questions

1. **Does class weighting consistently improve performance?**  
   **No.** Class weighting degraded Macro-F1 for pretrained CNNs while improving Macro-F1 for randomly initialized SwinUNETR.

2. **Does class weighting consistently improve LGG Recall?**  
   **No.** LGG Recall decreased significantly for ResNet18 (-33.3%) and DenseNet121 (-33.3%) under Protocol C, while increasing for SwinUNETR (+25.0%).

3. **Does stronger minority weighting (Protocol C) outperform Protocol B?**  
   **No for CNNs; Yes for SwinUNETR.** Protocol C caused severe LGG recall drops for ResNet18 ($0.6667 \rightarrow 0.4167$) and DenseNet121 ($0.5833 \rightarrow 0.3333$), but was required for SwinUNETR to achieve non-zero LGG detection.

4. **Does class weighting affect calibration?**  
   **Yes.** Class weighting distorts probability outputs. For ResNet18, Brier score doubled from 0.0496 (A) to 0.1019 (C).

5. **Does class weighting affect overconfidence safety?**  
   **Yes.** For ResNet18, high-confidence errors ($\ge 0.80$) increased from 1 error in Protocol A to 6 errors in Protocol B and C.

6. **What does this imply for imbalanced 3D medical imaging?**  
   Pretrained representations (e.g. MedicalNet 3D weights) already encode discriminative feature spaces that separate minority classes naturally. Imposing artificial loss re-weighting on rich feature spaces distorts the decision threshold and induces overconfident errors. Untrained architectures (SwinUNETR), however, require loss weighting to prevent majority-class collapse.

---

## 4. Final Cautionary Cautions
- **Do NOT claim**: "Class weighting is universally harmful" or "Protocol C is universally superior."
- **Correct Conclusion**: The utility of loss weighting is **strictly architecture-dependent** and interacts strongly with transfer learning initialization.
