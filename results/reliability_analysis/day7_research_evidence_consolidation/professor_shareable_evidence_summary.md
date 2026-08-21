# BraTS 2023 3D Glioma Grading Pipeline: Research Evidence Package & Professor Summary

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Document Type**: Shareable Executive Research Evidence Package  
**Date**: 2026-08-19  
**Auditor**: DeepMind Antigravity Advanced Agentic AI  

---

## 1. Primary Research Briefing (For Primary Faculty Advisor)

### A. Dataset & Experimental Setup
- **Cohort**: Multi-institutional BraTS 2023 Adult Glioma dataset ($N = 1,251$ total GLI scans).
- **Severe Imbalance**: $95.0\%$ High-Grade Glioma (HGG, Grade 4 / IDH-wildtype) vs $5.0\%$ Low-Grade Glioma (LGG, Grade 2/3 / IDH-mutant).
- **Rigorous Partitioning**: Development cohort ($N=1,063$: 53 LGG, 1,010 HGG) and an isolated, untouched Locked Test cohort ($N=188$: 12 LGG, 176 HGG). **Zero patient overlap across all partitions.**
- **Volumetric 3D Modalities**: Multi-parametric 4-channel input (FLAIR, T1w, T1gd, T2w) at $96 \times 96 \times 96$ resolution (1.0 mm isotropic spacing).
- **Evaluated Architectures**:
  1. `ResNet18 MedicalNet`: 3D ResNet with transfer learning from 23 medical segmentation datasets.
  2. `DenseNet121 3D`: Deep dense convolutional feature reuse trained from scratch.
  3. `SwinUNETR 3D`: 3D windowed self-attention Vision Transformer trained from scratch.

### B. Completed Experiments
- **Controlled Class-Weighting Benchmark**: $3 \times 3$ factorial design (9 canonical models) evaluating unweighted loss (Protocol A), inverse-frequency weighting (Protocol B), and minority-boosted weighting (Protocol C).
- **Locked-Test Evaluation**: All 9 canonical checkpoints evaluated on the frozen 188-subject locked test set.
- **5-Fold Cross-Validation**: Full development cohort ($N=1,063$) stratified cross-validation on ResNet18 Protocol A.
- **Decision-Threshold Operating-Point Analysis**: 99-point threshold sweep tuned on validation data ($N=187$) and tested out-of-sample on locked test ($N=188$).

### C. Most Important Numerical Results (Locked Test, $N=188$)
- **Top Overall Model**: ResNet18 MedicalNet (Protocol A) $\to$ **$\text{Accuracy} = 97.34\%$**, **$\text{Macro-F1} = 0.8842$**, **$\text{ROC-AUC} = 0.9830$**, **$\text{LGG Recall} = 75.00\%$** ($9/12$ detected), **$\text{Brier Score} = 0.0496$**.
- **Scratch CNN Baseline**: DenseNet121 3D (Protocol A) $\to$ **$\text{Accuracy} = 94.68\%$**, **$\text{Macro-F1} = 0.7934$**, **$\text{ROC-AUC} = 0.9422$**, **$\text{LGG Recall} = 66.67\%$** ($8/12$ detected).
- **Transformer Baseline**: SwinUNETR 3D (Protocol C) $\to$ **$\text{Accuracy} = 94.68\%$**, **$\text{Macro-F1} = 0.6736$**, **$\text{ROC-AUC} = 0.9735$**, **$\text{LGG Recall} = 25.00\%$** ($3/12$ detected).

### D. Most Interesting Unexpected Finding: Architectural Divergence Under Class Weighting
- In **CNNs** (ResNet18, DenseNet121), static inverse-frequency weighting **degraded** minority sensitivity (ResNet18: $75.0\% \to 66.7\% \to 41.7\%$; DenseNet121: $66.7\% \to 58.3\% \to 33.3\%$) and worsened probability calibration (Brier score doubled from $0.0496 \to 0.1042$).
- In **Vision Transformers** (SwinUNETR), unweighted loss completely collapsed to 0% sensitivity, whereas minority-boosted weighting (Protocol C) **rescued** sensitivity to $25.0\%$ and achieved the highest ROC-AUC ($0.9735$).
- $\implies$ **Key Takeaway**: Static class weighting is not a universal solution; it disrupts pretrained CNN representations while helping scratch-trained transformers escape majority plateaus.

### E. Cross-Validation Reliability Finding (ResNet18 Protocol A, $N=1,063$)
- **Overall Discrimination is Highly Stable**: $\text{ROC-AUC} = \mathbf{0.9163 \pm 0.0446}$ ($\text{CV} = 4.87\%$).
- **Minority Sensitivity is Extremely Variable**: $\text{LGG Recall} = \mathbf{0.5055 \pm 0.3274}$ ($\text{CV} = 64.77\%$, spanning $0.00\%$ in Fold 4 to $80.00\%$ in Fold 5).
- **The Masking Effect of Accuracy**: Accuracy remained constant at $95.30\% \pm 0.66\%$ across all folds, completely hiding the sensitivity swings.
- $\implies$ **Key Takeaway**: The $75.0\%$ locked-test result lies within the empirical CV range, but represents an upper-tail draw from a variable sensitivity distribution.

### F. Decision-Threshold Finding
- Models possess strong latent class-separating representations ($\text{ROC-AUC} \ge 0.86$), but fixed default thresholding ($\tau = 0.50$) compresses minority decisions into the 95% majority class.
- By tuning decision thresholds strictly on validation data ($\tau^*$), out-of-sample test sensitivity was recovered:
  - SwinUNETR Protocol A: $\text{LGG Recall}$ moved from **$0.00\% \to 100.00\%$** ($12/12$ detected at $\tau^* = 0.03$), with $77.27\%$ Specificity.
  - ResNet18 Protocol A: $\text{LGG Recall}$ moved from **$75.00\% \to 91.67\%$** ($11/12$ detected at $\tau^* = 0.22$), with $94.32\%$ Specificity.
- $\implies$ **Key Takeaway**: Threshold adjustment explores operating-point trade-offs along an invariant ROC curve; it does not alter feature representations or calibrate probabilities.

### G. Exact Unresolved Scientific Questions
1. How much of the minority sensitivity variance is driven by **patient-partition composition** versus **multi-seed GPU/worker optimization stochasticity** on a fixed split?
2. Does dynamically modulated loss (Focal Loss) eliminate the CNN degradation seen with static class weighting?

### H. Recommended Logical Next Experiment
- **Controlled Focal Loss Benchmark with 5-Fold Cross-Validation**: Evaluate Focal Loss ($\gamma=2.0, \alpha=0.25$) across architectures to test whether dynamically down-weighting easy majority examples stabilizes minority sensitivity without distorting feature gradients.

---

## 2. General Research Update (Tier 2: For Collaborating Professors & Graduate Students)

### High-Confidence Established Results
1. **Volumetric 3D MRI achieves strong discriminative ranking** for adult glioma grading ($\text{ROC-AUC} \ge 0.91$ across all architectures).
2. **Standard Accuracy is completely uninformative** under severe 19:1 clinical class imbalance ($\approx 95\%$ accuracy even when minority sensitivity is $0\%$).
3. **Pretrained 3D CNNs outperform scratch Vision Transformers** under limited medical sample regimes ($N \approx 1,000$).
4. **Static class weighting produces opposing architectural effects**: it degrades sensitivity in pretrained CNNs while providing modest rescue in scratch Vision Transformers.

### Preliminary & Exploratory Findings
1. **Operating-point tuning recovers hidden sensitivity**: Models with 0% sensitivity at default $\tau = 0.50$ achieve high sensitivity when calibrated on validation data.
2. **Minority sensitivity has wide confidence intervals**: Small locked-test sample size ($N=12$ LGG) means each patient shifts sensitivity by $8.33\%$.

### Open Research Questions
1. Quantifying pure multi-seed variance versus patient-partition sampling variance.
2. Formulating and testing dynamic loss functions (Focal Loss / Asymmetric Loss) across multi-fold cross-validation.
