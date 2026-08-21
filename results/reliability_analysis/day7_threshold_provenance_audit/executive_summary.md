# Day 7 — Audit 6: Decision-Threshold Analysis Executive Summary

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Investigation**: Phase 7 Decision-Threshold Sensitivity & Operating-Point Analysis Provenance  
**Audit Type**: Strict Read-Only Forensic & Methodological Audit  
**Date**: 2026-08-19  

---

## A. THRESHOLD PIPELINE PROVENANCE

1. **Strict Validation-Only Tuning**:
   - Threshold search grid (99 thresholds, $\tau \in [0.01, 0.99]$) was evaluated strictly on the $N=187$ development validation cohort (10 LGG, 177 HGG).
   - Selection rules (Rule A: Max Macro-F1; Rule B: Max LGG Recall s.t. Specificity $\ge 90\%$; Rule C: Max Balanced Accuracy) selected optimal threshold cutoffs $\tau^*$ exclusively from validation metrics.
2. **Frozen Out-of-Sample Test Evaluation**:
   - The validation-derived threshold $\tau^*$ was frozen before applying it to the untouched 188-subject locked test set (12 LGG, 176 HGG).

---

## B. TEST-SET CONTAMINATION ASSESSMENT

1. **Zero Test-Set Contamination (VERIFIED ABSENT)**:
   - Locked-test labels, predictions, and metrics were never accessed or evaluated during the threshold search or selection phase.
2. **Preservation of Canonical Results**:
   - Canonical locked-test results at default $\tau = 0.50$ remain isolated and untouched in `results/locked_test/`. Threshold results are strictly isolated under `results/reliability_analysis/threshold_analysis/`.

---

## C. VALIDATION $\to$ TEST GENERALIZATION

1. **Significant Sensitivity Recovery Across Models**:
   - **SwinUNETR Protocol A**: Validation-tuned $\tau^* = 0.03$ recovered test LGG Recall from **$0.00\% \to 100.00\%$** ($12/12$ detected), with HGG Specificity moving from $100.0\% \to 77.27\%$.
   - **ResNet18 Protocol A**: Validation-tuned $\tau^* = 0.22$ increased test LGG Recall from **$75.00\% \to 91.67\%$** ($11/12$ detected), while preserving $94.32\%$ HGG Specificity ($166/176$ correct).
   - **SwinUNETR Protocol C**: Validation-tuned $\tau^* = 0.18$ increased test LGG Recall from **$25.00\% \to 83.33\%$** ($10/12$ detected), while preserving $95.45\%$ HGG Specificity.

---

## D. WHAT THRESHOLD ANALYSIS PROVES

1. **Presence of Latent Discriminative Capacity**:
   - High ROC-AUC values ($\ge 0.86$ across models) indicate that networks learn separable latent feature representations, but default argmax thresholding ($\tau = 0.50$) compresses hard predictions into the 95% majority class.
2. **Tunability of Operating Points**:
   - Operating points can be systematically calibrated on validation data to achieve clinically targeted sensitivity/specificity trade-offs.

---

## E. WHAT IT DOES NOT PROVE

1. **Does NOT Improve Model Representation**:
   - ROC-AUC is mathematically invariant to threshold shifts; threshold tuning moves along the existing ROC curve rather than expanding it.
2. **Does NOT Constitute Probability Calibration**:
   - Threshold optimization does not adjust predicted probability distributions (ECE and Brier score remain unchanged).
3. **Does NOT Prove Clinical Superiority**:
   - Sensitivity gains come at the cost of increased majority-class false positives ($100\%$ LGG sensitivity in SwinUNETR resulted in $40$ false positives on HGG).

---

## F. REPORTING-READY LANGUAGE

- **Recommended Paper Formulation**:
  > *"Primary model performance is reported on the locked test set ($N=188$) using frozen checkpoints and the default decision threshold ($\tau = 0.50$). In secondary exploratory sensitivity analysis, decision thresholds were calibrated strictly on development validation data to evaluate alternative clinical operating points. For ResNet18 (Protocol A), a validation-derived threshold of $\tau^* = 0.22$ achieved $91.67\%$ LGG Recall ($11/12$) with $94.32\%$ HGG Specificity. For SwinUNETR (Protocol A), $\tau^* = 0.03$ recovered LGG Recall from $0.00\%$ to $100.00\%$ with $77.27\%$ Specificity, confirming that the model learns discriminative representations ($\text{ROC-AUC} = 0.9337$) that are obscured by default argmax thresholding."*

---

## G. UNRESOLVED QUESTIONS

1. **Multi-Fold Threshold Stability**:
   - The stability and variance of validation-derived $\tau^*$ across 5-fold cross-validation partitions remains unquantified.
