# Day 7 — Audit 2: LGG Recall Anomaly Executive Summary

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Investigation**: Minority-Class (LGG) Recall Instability & Failure Mechanisms  
**Audit Type**: Strict Read-Only Forensic Audit  
**Date**: 2026-08-19  

---

## A. ESTABLISHED FACTS

1. **Zero Data Leakage or Label Inconsistency**:
   - Pairwise patient overlap across training ($N=876$), validation ($N=187$), locked-test ($N=188$), and 5-fold cross-validation partitions ($N=1,063$) is strictly **0** (Proven Absent).
   - Label encoding (LGG=0, HGG=1) is identical across all scripts, manifests, models, and prediction files.
2. **Root Cause of Historical 0.00 vs 0.75 Discrepancy**:
   - The earlier 0.00 LGG recall result came from an **early 7-epoch checkpoint** where early stopping prematurely triggered at Epoch 7 and restored Epoch 1 weights before minority representations separated.
   - The canonical 0.75 LGG recall result came from the **canonical 25-epoch checkpoint** where training continued to Epoch 25 and restored optimal Epoch 19 weights ($\text{Val Macro-F1} = 0.7642, \text{Val LGG Recall} = 0.7000$).
   - Both results were evaluated using identical preprocessing, metric formulas, and default threshold $\tau = 0.5$.
3. **High Empirical Cross-Validation Variance**:
   - Across 5-fold cross-validation on the 1,063-subject development cohort, ResNet18 Protocol A achieved $\text{Mean LGG Recall} = 0.5055 \pm 0.3274$ (Fold range: 0.0000 to 0.8000).
   - Majority-class (HGG) recall remained stable at $0.9762 \pm 0.0196$.
4. **Stable Latent Discriminative Capacity (ROC-AUC)**:
   - ResNet18 MedicalNet maintains high continuous discriminative ability across all partitions ($\text{Mean ROC-AUC} = 0.9163 \pm 0.0446$ in CV, and $0.9830$ on locked test).
   - Even in Fold 4, where fixed-threshold recall collapsed to 0.00%, ROC-AUC remained high at 0.8629.
5. **Minority-Class Denominator Constraint**:
   - The locked test set contains only 12 LGG cases, and validation folds contain only 10–11 LGG cases. Each individual patient classification shifts sensitivity by $8.33\%\text{--}10.00\%$.

---

## B. STRONGLY SUPPORTED INTERPRETATIONS

1. **Patient-Distribution Composition as a Major Sensitivity Driver**:
   - Stratified cross-validation demonstrates that sensitivity is highly sensitive to the specific patient composition of the validation split (ranging from 0% to 80% under identical code and seeds).
2. **Decision-Threshold Collapse Under Severe 19:1 Imbalance**:
   - The fixed default threshold $\tau = 0.5$ fails under unweighted empirical cross-entropy loss because predicted probabilities for minority cases are compressed toward zero, causing thresholded sensitivity to collapse even when continuous ranking (ROC-AUC) remains strong.
3. **Canonical 0.75 Result Lies in the Upper Tail of the Performance Envelope**:
   - The 0.75 locked-test result was accurately anticipated by its validation checkpoint (0.70 at Epoch 19), but represents an outcome in the upper performance tier ($0.5055 \pm 0.3274$) rather than a guaranteed invariant.
4. **ResNet18 Representation Sufficiency**:
   - ResNet18 with MedicalNet pretraining has sufficient representational capacity to grade 3D gliomas; the bottleneck lies in objective function optimization and threshold calibration under extreme class imbalance.

---

## C. UNRESOLVED QUESTIONS

1. **Pure Optimization Stochasticity vs Patient-Sampling Variance**:
   - Current evidence confirms that patient distribution drives substantial variance across folds. However, the exact proportion of variance attributable to multi-seed training stochasticity on a single fixed split remains unquantified without a dedicated multi-seed ablation experiment.
2. **Deterministic cuDNN / Worker Seed Impact**:
   - While top-level seed 42 was applied, non-deterministic cuDNN benchmark algorithms and multiprocessing worker augmentation streams were active, leaving open the exact contribution of GPU-level non-determinism to early-stopping trajectories.
