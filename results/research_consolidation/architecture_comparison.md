# Architecture Comparison: ResNet18 vs DenseNet121 vs SwinUNETR

This document synthesizes how neural architecture interacts with class imbalance, pretraining, calibration, and reliability within the evaluated configurations.

---

## 1. Summary Comparison Table

| Architecture | Best Protocol | Best Macro-F1 | Best LGG Recall | Best AUC | Best ECE | Mean Macro-F1 | Mean LGG Recall | Mean ECE | Total High-Conf Errors ($\ge 0.80$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet18 MedicalNet** | **A** | **0.8842** | **0.7500** | **0.9830** | **0.0276** | 0.7664 | 0.6111 | 0.0271 | 13 |
| **DenseNet121 3D** | **A** | **0.7934** | **0.6667** | **0.9422** | **0.0308** | 0.7188 | 0.5278 | 0.0400 | 13 |
| **SwinUNETR 3D** | **C** | **0.6736** | **0.2500** | **0.9735** | **0.0359** | 0.5469 | 0.0833 | 0.0250 | 28 |

---

## 2. Key Synthesis Findings

1. **ResNet18 MedicalNet (Pretrained 3D CNN)**:
   - **Performance**: Achieved the top overall performance in the evaluated pipeline (**Macro-F1: 0.8842**, **LGG Recall: 0.7500**, **AUC: 0.9830** under Protocol A).
   - **Pretraining Advantage**: Initializing with MedicalNet 3D weights provided a structured spatial representation that naturally separated LGG and HGG features without loss weighting.
   - **Sensitivity to Class Weighting**: Highly sensitive to loss weighting; Protocol C degraded LGG Recall from 0.7500 to 0.4167.

2. **DenseNet121 3D (Dense Feature Reuse CNN)**:
   - **Performance**: Solid secondary baseline (**Macro-F1: 0.7934**, **LGG Recall: 0.6667** under Protocol A).
   - **Feature Reuse**: Dense connectivity provided strong feature propagation, but unweighted Protocol A still outperformed weighted Protocols B and C.

3. **SwinUNETR 3D (Vision Transformer Initialized from Scratch)**:
   - **Performance**: Showed majority-class collapse under Protocol A & B (**0.0% LGG Recall**).
   - **Pretraining Requirement**: Lacking 3D pretrained transformer weights, SwinUNETR required Protocol C weighting to enable non-zero minority detection (**LGG Recall: 0.2500**, **Macro-F1: 0.6736**).
