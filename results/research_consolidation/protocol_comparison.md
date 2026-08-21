# Protocol Comparison: Unweighted (A) vs Inverse-Frequency (B) vs Strong Minority (C)

This document formalizes the pairwise protocol transitions ($A \to B$, $B \to C$, $A \to C$) across all three architectures based on Phase 4 ablation deltas.

---

## 1. Master Pairwise Transition Delta Table

| Architecture | Transition | $\Delta$ Macro-F1 | Macro-F1 Effect | $\Delta$ LGG Recall | LGG Recall Effect | $\Delta$ AUC | $\Delta$ ECE | $\Delta$ Brier | $\Delta$ High-Conf Err Rate | $\Delta R_B$ |
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

## 2. Main Protocol Insights
1. **Unweighted Protocol A**: Optimal for pretrained CNNs (ResNet18 & DenseNet121).
2. **Canonical Inverse-Frequency Protocol B**: Showed mild performance loss for CNNs without improving LGG recall.
3. **Strong Minority Weighting Protocol C**: Detrimental to pretrained CNNs (-33.3% LGG recall), but beneficial for untrained SwinUNETR (+25.0% LGG recall).
