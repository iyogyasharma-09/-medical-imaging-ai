# Day 7 — Audit 3: Controlled Class-Weighting Effect Executive Summary

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Investigation**: Experimental Validity, Within-Architecture Effect Sizes, and Scientific Interpretation of Protocols A, B, and C  
**Audit Type**: Strict Read-Only Forensic & Statistical Interpretation Audit  
**Date**: 2026-08-19  

---

## 1. What the Experiment Proves (Established Facts)

1. **Class Weighting Was a Rigorously Controlled Experimental Variable**:
   - Protocols A, B, and C strictly isolated class weighting ($w=\text{None}$, $w=[1.9018, 0.0982]$, $w=[1.9207, 0.0793]$, all with sum = 2.0) across all 3 architectures while holding dataset splits, seed (42), preprocessing, augmentations, optimizer, scheduler, and decision threshold (0.5) constant.
2. **Class Weighting Did NOT Universally Improve Minority-Class Sensitivity**:
   - In ResNet18 MedicalNet, LGG Recall declined monotonically: $\text{Prot A } (75.0\%) \to \text{Prot B } (66.7\%) \to \text{Prot C } (41.7\%)$.
   - In DenseNet121 3D, LGG Recall declined monotonically: $\text{Prot A } (66.7\%) \to \text{Prot B } (58.3\%) \to \text{Prot C } (33.3\%)$.
   - In SwinUNETR 3D, sensitivity remained at $0.0\%$ under Protocols A and B, but improved to $25.0\%$ under Protocol C.
3. **Accuracy is Completely Uninformative Under Severe Imbalance**:
   - Across all 9 configurations, overall test accuracy remained bounded between $91.49\%$ and $97.34\%$, masking severe fluctuations in minority-class sensitivity ($0.0\%$ to $75.0\%$).
4. **Continuous Latent Discrimination (ROC-AUC) Remains High and Stable**:
   - All 9 models achieved ROC-AUC between $0.8643$ and $0.9830$, proving that models learned separable latent representations even when thresholded classification collapsed at $\tau = 0.5$.

---

## 2. What the Experiment Strongly Suggests (Supported Interpretations)

1. **Architecture-Dependent Empirical Response on This Benchmark**:
   - The empirical data demonstrates divergent architectural behavior: static class weighting degraded performance in CNNs (ResNet18, DenseNet121) while providing modest sensitivity recovery in the Vision Transformer (SwinUNETR).
2. **Fixed Thresholding ($\tau = 0.5$) Exacerbates Sensitivity Collapse**:
   - Post-hoc threshold calibration reveals that high sensitivity ($100\%$ in SwinUNETR Protocol A) can be unlocked by shifting the decision boundary, proving that poor sensitivity is an interaction between uncalibrated posterior probabilities and fixed argmax thresholding.
3. **Unweighted Protocol A ResNet18 is the Empirical Top Performer on This Split**:
   - Protocol A ResNet18 achieved the highest Macro-F1 ($0.8842$), LGG Recall ($0.7500$), and ROC-AUC ($0.9830$), with the lowest Brier score ($0.0496$).

---

## 3. What Remains Uncertain (Unresolved Questions & Caveats)

1. **Small Minority Sample Size Denominator ($N=12$)**:
   - Because 1 LGG subject equals $8.333\%$ of sensitivity, adjacent 1-case shifts (e.g., $9/12 \to 8/12$) carry overlapping $95\%$ confidence intervals ($[46.8\%, 91.1\%]$ vs $[39.1\%, 86.2\%]$) and should be interpreted as empirical observations rather than population-level statistical proofs.
2. **Generalizability of CV Variance to Protocols B and C**:
   - Phase 6 Cross-Validation was performed exclusively on ResNet18 Protocol A ($0.5055 \pm 0.3274$). The variance profile of Protocols B and C across multiple folds remains unmeasured.
3. **Decoupling Pretraining vs Architectural Inductive Bias**:
   - Whether ResNet18's degradation under weighting was caused by MedicalNet pretraining or CNN convolutional inductive bias remains an unresolved hypothesis requiring a scratch-trained ResNet18 baseline.
