# Day 7 — Audit 5: Cross-Validation Scope & Reliability Executive Summary

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Investigation**: Phase 6 Five-Fold Cross-Validation Scope, Metric Reliability, and Generalizability  
**Audit Type**: Strict Read-Only Forensic & Statistical Reliability Audit  
**Date**: 2026-08-19  

---

## A. ESTABLISHED CV FACTS

1. **Experimental Design & Partition Integrity**:
   - Phase 6 cross-validation evaluated the full 1,063-subject development cohort (53 LGG, 1,010 HGG; 4.99% minority prevalence) across 5 stratified folds.
   - Every development subject appeared in validation **exactly once** across the 5 folds (100% coverage, 0 train/val overlap, 0 locked-test overlap).
2. **Aggregated Cross-Validation Metrics (ResNet18 MedicalNet Protocol A)**:
   - **Overall Accuracy**: $0.9530 \pm 0.0066$ (Range: $0.9437\text{--}0.9624$; $\text{CV} = 0.69\%$)
   - **HGG Recall (Specificity)**: $0.9762 \pm 0.0196$ (Range: $0.9604\text{--}1.0000$; $\text{CV} = 2.01\%$)
   - **ROC-AUC (Discrimination)**: $0.9163 \pm 0.0446$ (Range: $0.8629\text{--}0.9634$; $\text{CV} = 4.87\%$)
   - **Macro-F1 Score**: $0.7145 \pm 0.1290$ (Range: $0.4879\text{--}0.7952$; $\text{CV} = 18.06\%$)
   - **LGG Recall (Sensitivity)**: $0.5055 \pm 0.3274$ (Range: $0.0000\text{--}0.8000$; $\text{CV} = 64.77\%$)

---

## B. RELIABILITY FINDINGS

1. **Dichotomy Between Latent Representation and Thresholded Decisions**:
   - Continuous discriminative ranking (ROC-AUC) is stable across all patient partitions ($\text{CV} = 4.87\%$).
   - Hard classification decisions at default threshold $\tau = 0.50$ are extremely unstable ($\text{CV} = 64.77\%$), swinging from complete failure in Fold 4 ($0/10$ detected) to strong sensitivity in Fold 5 ($8/10$ detected).
2. **The Masking Effect of Overall Accuracy**:
   - Accuracy remains steady at ~95.3% across all folds, completely masking whether the model detects $0\%$ or $80\%$ of minority cases.

---

## C. GENERALIZABILITY LIMITS

1. **Scope Restriction**:
   - These cross-validation findings apply strictly to **ResNet18 MedicalNet under Protocol A** on the BraTS 2023 development cohort.
2. **Prohibited Generalizations**:
   - Do **NOT** extrapolate this variance profile to DenseNet121, SwinUNETR, Protocol B, or Protocol C without empirical 5-fold cross-validation runs.
   - Do **NOT** claim that Focal Loss is proven to resolve this inter-fold variance without experimental proof.

---

## D. UNRESOLVED QUESTIONS

1. **Partitioning Variance vs Optimization Stochasticity**:
   - The exact quantitative split between patient-distribution variance and multi-seed optimization trajectory variance remains unmeasured on a fixed split.
2. **Multi-Architecture Cross-Validation Profile**:
   - The cross-validation performance envelope of DenseNet121 and SwinUNETR remains untested under 5-fold cross-validation.

---

## E. REPORTING-READY LANGUAGE

- **Recommended Formulation**:
  > *"Under 5-fold cross-validation on the development cohort ($N=1,063$), ResNet18 MedicalNet with unweighted cross-entropy loss demonstrated stable overall discrimination ($\text{ROC-AUC} = 0.9163 \pm 0.0446$; $\text{Accuracy} = 95.30\% \pm 0.66\%$), but exhibited substantial fold-to-fold sensitivity variability on the minority class ($\text{LGG Recall} = 0.5055 \pm 0.3274$, range: $0.0000\text{--}0.8000$) at the default classification threshold ($\tau = 0.50$). The locked-test result of $75.0\%$ lies within the observed cross-validation range but above its mean."*
