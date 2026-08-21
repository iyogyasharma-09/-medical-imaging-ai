# Day 7 — Audit 6: Decision-Threshold Analysis Provenance & Methodological Validity Audit

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Audit Date**: 2026-08-19  
**Audit Mode**: Strict Read-Only Forensic & Methodological Audit  
**Auditor**: DeepMind Antigravity Advanced Agentic AI  
**Scope**: Phase 7 Operating-Point Sensitivity Analysis, Validation Tuning, Test-Set Isolation, and Reporting Validity  

---

## 1. Executive Summary & Central Question

### 1.1 Purpose of Audit
In the BraTS 2023 3D glioma grading pipeline, performance has been reported under two distinct modes:
1. **Canonical Locked-Test Evaluation**: Frozen checkpoints evaluated at fixed default threshold $\tau = 0.50$ on the 188-subject locked test set.
2. **Phase 7 Threshold Sensitivity & Operating-Point Analysis**: Decision thresholds $\tau^*$ tuned strictly on the 187-subject development validation set and subsequently applied out-of-sample to the locked test set.

The central question addressed by **Audit 6** is:
> *"Was the threshold analysis performed with a methodologically valid train/validation/test separation, and what exactly can be claimed from the resulting threshold adjustments?"*

### 1.2 Central Forensic Finding
- **Zero Test Contamination**: Threshold search was conducted **strictly on validation probabilities** ($N=187$: 10 LGG, 177 HGG) without test-set label exposure (**VERIFIED ABSENT**).
- **Legitimate Out-of-Sample Operating-Point Evaluation**: The chosen $\tau^*$ was frozen before evaluating locked-test probabilities ($N=188$: 12 LGG, 176 HGG).
- **Scientific Nature of Threshold Adjustment**: Threshold optimization shifts the classification operating point along an invariant ROC curve; it **does not improve model representation** (ROC-AUC is unchanged) and **does not constitute probability calibration** (Brier score and ECE are unchanged). It represents an exploratory operating-point sensitivity analysis.

---

## 2. Part 1 — Inventory of Phase 7 Threshold Artifacts

All 12 artifacts located in `results/reliability_analysis/threshold_analysis/` were audited and verified:

| Filename | File Type | Size (Bytes) | SHA-256 (Prefix) | Methodological Purpose | Provenance Status |
| :--- | :---: | :---: | :--- | :--- | :---: |
| `threshold_sweep_results.csv` | Dataset | 127,691 | `2d7a96a1...` | Full 99-point threshold sweep across all 9 models on validation set | **VERIFIED** |
| `threshold_selected_results.csv` | Dataset | 2,191 | `15e58129...` | Validation-selected $\tau^*$ and out-of-sample test results for Rules A, B, C | **VERIFIED** |
| `threshold_comparison.csv` | Dataset | 1,395 | `2f9dca12...` | Comparison of default $\tau=0.50$ vs validation-selected $\tau^*$ on test set | **VERIFIED** |
| `threshold_analysis_report.md` | Report | 9,773 | `91e4ab6e...` | Formal Phase 7 technical report and clinical utility analysis | **VERIFIED** |
| `threshold_methodology.md` | Report | 4,089 | `b6acff23...` | Mathematical specification of candidate threshold selection rules | **VERIFIED** |
| `threshold_interpretation.md` | Report | 5,972 | `e685f0ef...` | Clinical trade-off analysis and decision boundary interpretation | **VERIFIED** |
| `threshold_analysis_manifest.json` | Manifest | 2,728 | `9d555c82...` | Cryptographic SHA-256 signature manifest for Phase 7 artifacts | **VERIFIED** |
| `roc_operating_points.png` | Plot | 225,598 | `4b09e25d...` | ROC curves depicting default vs selected operating points | **VERIFIED** |
| `lgg_recall_vs_threshold.png` | Plot | 178,346 | `3b8d4f40...` | LGG sensitivity curves across threshold domain $\tau \in [0.01, 0.99]$ | **VERIFIED** |
| `specificity_vs_threshold.png` | Plot | 175,793 | `5f0d3a51...` | HGG specificity curves across threshold domain | **VERIFIED** |
| `macro_f1_vs_threshold.png` | Plot | 186,508 | `78e8b08f...` | Macro-F1 curves across threshold domain | **VERIFIED** |
| `balanced_accuracy_vs_threshold.png` | Plot | 193,330 | `95a7c29e...` | Balanced accuracy (Youden's J) curves across threshold domain | **VERIFIED** |

---

## 3. Part 2 — Reconstruction of the Threshold-Selection Pipeline

Inspection of `run_threshold_analysis.py` confirms the following strict sequential workflow:

```
[Frozen Checkpoints (Day 5 Archives)]
                  ↓
[Validation Forward Pass (N=187: 10 LGG, 177 HGG)]
                  ↓
[Generate Validation Probabilities (p_lgg, p_hgg)]
                  ↓
[Threshold Grid Sweep: 99 Thresholds, tau in [0.01, 0.99]]
                  ↓
[Evaluate Selection Rules on Validation Data Only]
   • Rule A: Maximize Validation Macro-F1
   • Rule B: Maximize Validation LGG Recall s.t. HGG Specificity >= 90%
   • Rule C: Maximize Validation Balanced Accuracy (Youden's J)
                  ↓
[Freeze Selected Optimal Thresholds (tau*)]
                  ↓
[Load Frozen Locked-Test Probabilities (N=188: 12 LGG, 176 HGG)]
                  ↓
[Apply Frozen tau* Out-of-Sample to Locked-Test Set]
                  ↓
[Output Operating-Point Comparison Tables & Plots]
```

**Verification**: At no point during grid search or objective optimization were locked-test labels or probabilities accessed.

---

## 4. Part 3 — Test-Set Contamination Assessment

| Investigation Dimension | Audit Question | Forensic Evidence | Verdict |
| :--- | :--- | :--- | :---: |
| **Threshold Search Domain** | Were locked-test labels used to select $\tau^*$? | Grid search evaluated strictly on `val_data["y_true"]` ($N=187$); $\tau^*$ was chosen from validation summary table `df_v_sweep`. | **VERIFIED ABSENT** |
| **Model / Checkpoint Selection** | Were checkpoints swapped based on threshold performance? | Checkpoints loaded from `protocol_archives/` with SHA-256 signatures identical to Day 5 archives. | **VERIFIED ABSENT** |
| **Protocol Selection** | Were Protocols A/B/C chosen based on threshold tuning? | All 9 models across all 3 protocols were systematically audited under identical rules. | **VERIFIED ABSENT** |
| **Canonical Metric Integrity** | Were canonical results in `locked_test/` overwritten? | `results/locked_test/` files maintain default $\tau = 0.50$; threshold results are isolated in `reliability_analysis/threshold_analysis/`. | **VERIFIED ABSENT** |
| **Temporal Sequence** | Was threshold analysis prespecified or post-hoc? | Threshold analysis was designed and executed on Day 6 as Phase 7 exploratory sensitivity analysis after observing canonical results. | **NO EVIDENCE OF CONTAMINATION (EXPLORATORY ANALYSIS)** |

---

## 5. Part 4 — Default $\tau = 0.50$ vs Validation-Selected $\tau^*$ (Complete 9-Model Matrix)

The table below contrasts default locked-test performance against validation-selected operating points:

| Protocol | Model | Test AUC | Default $\tau = 0.50$ | Default LGG Rec | Default HGG Spec | Default Macro-F1 | Selected $\tau^*$ (Rule B) | Rule B LGG Rec | Rule B HGG Spec | Rule B Macro-F1 | Selected $\tau^*$ (Rule C) | Rule C LGG Rec | Rule C HGG Spec | Rule C Macro-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **ResNet18** | **0.9830** | 0.50 | **0.7500** (9/12) | 0.9886 | **0.8842** | 0.23 | **0.8333** (10/12) | 0.9489 | 0.8066 | 0.22 | **0.9167** (11/12) | 0.9432 | 0.8173 |
| **A** | **DenseNet121** | **0.9422** | 0.50 | **0.6667** (8/12) | 0.9659 | **0.7934** | 0.27 | **0.6667** (8/12) | 0.9432 | 0.7464 | 0.05 | **1.0000** (12/12) | 0.7443 | 0.6006 |
| **A** | **SwinUNETR** | **0.9337** | 0.50 | **0.0000** (0/12) | 1.0000 | **0.4835** | 0.04 | **0.0000** (0/12) | 1.0000 | 0.4835 | 0.03 | **1.0000** (12/12) | 0.7727 | 0.6234 |
| **B** | **ResNet18** | **0.8939** | 0.50 | **0.6667** (8/12) | 0.9318 | **0.7267** | 0.62 | **0.6667** (8/12) | 0.9716 | 0.8072 | 0.62 | **0.6667** (8/12) | 0.9716 | 0.8072 |
| **B** | **DenseNet121** | **0.9460** | 0.50 | **0.5833** (7/12) | 0.9375 | **0.7102** | 0.40 | **0.7500** (9/12) | 0.9148 | 0.7235 | 0.11 | **1.0000** (12/12) | 0.7557 | 0.6095 |
| **B** | **SwinUNETR** | **0.8643** | 0.50 | **0.0000** (0/12) | 1.0000 | **0.4835** | 0.06 | **0.2500** (3/12) | 0.9943 | 0.6736 | 0.06 | **0.2500** (3/12) | 0.9943 | 0.6736 |
| **C** | **ResNet18** | **0.9167** | 0.50 | **0.4167** (5/12) | 0.9602 | **0.6884** | 0.52 | **0.4167** (5/12) | 0.9602 | 0.6884 | 0.08 | **0.8333** (10/12) | 0.8068 | 0.6223 |
| **C** | **DenseNet121** | **0.9013** | 0.50 | **0.3333** (4/12) | 0.9602 | **0.6527** | 0.49 | **0.3333** (4/12) | 0.9602 | 0.6527 | 0.06 | **0.9167** (11/12) | 0.7159 | 0.5665 |
| **C** | **SwinUNETR** | **0.9735** | 0.50 | **0.2500** (3/12) | 0.9943 | **0.6736** | 0.24 | **0.6667** (8/12) | 0.9659 | 0.7934 | 0.18 | **0.8333** (10/12) | 0.9545 | 0.8189 |

---

## 6. Part 5 & 6 — Mathematical Definition of Selection Rules (Rule B & Rule C)

### 6.1 Rule A: Maximize Validation Macro-F1
$$\tau_A^* = \arg\max_{\tau \in [0.01, 0.99]} \text{Macro-F1}_{\text{val}}(\tau)$$
- **Tie-Breaker**: Highest LGG Recall, then highest Balanced Accuracy.

### 6.2 Rule B: High-Sensitivity Clinical Operating Point
$$\tau_B^* = \arg\max_{\tau \in \mathcal{T}_{\ge 0.90}} \text{LGG-Recall}_{\text{val}}(\tau), \quad \text{where } \mathcal{T}_{\ge 0.90} = \{ \tau \mid \text{HGG-Recall}_{\text{val}}(\tau) \ge 0.90 \}$$
- **Constraint**: Strict preservation of $\ge 90\%$ specificity on majority class.
- **Tie-Breaker**: Highest Macro-F1.

### 6.3 Rule C: Maximize Balanced Accuracy (Youden's J Index)
$$\tau_C^* = \arg\max_{\tau \in [0.01, 0.99]} \left( \frac{\text{LGG-Recall}_{\text{val}}(\tau) + \text{HGG-Recall}_{\text{val}}(\tau)}{2} \right)$$
- **Objective**: Finds the optimal cutoff balancing true positive and true negative rates equally.

**Methodological Verdict**: Rules A, B, and C represent **legitimate, validation-derived operating-point selection**, completely free from test-set optimization.

---

## 7. Part 7 & 8 — Validation $\to$ Test Generalization & Small Sample Effect ($N=12$)

### 7.1 Key Generalization Dynamics
1. **SwinUNETR Protocol A**: Validation tuning identified $\tau^* = 0.03$. On the locked test, this shifted LGG Recall from **$0.00\% \to 100.00\%$** ($12/12$ detected), with HGG Specificity moving from $100.0\% \to 77.27\%$.
2. **ResNet18 Protocol A**: Rule C identified $\tau^* = 0.22$. On the locked test, LGG Recall increased from **$75.00\% \to 91.67\%$** ($11/12$ detected), while preserving $94.32\%$ specificity ($166/176$ HGG correct).
3. **SwinUNETR Protocol C**: Rule C identified $\tau^* = 0.18$. On the locked test, LGG Recall increased from **$25.00\% \to 83.33\%$** ($10/12$ detected), while preserving $95.45\%$ specificity.

### 7.2 Patient-Level Granularity on Locked Test ($N=12$)
- $75.00\% \to 91.67\%$: Exactly **$+2$ additional LGG patients detected** ($9/12 \to 11/12$).
- $25.00\% \to 83.33\%$: Exactly **$+7$ additional LGG patients detected** ($3/12 \to 10/12$).
- $0.00\% \to 100.00\%$: Exactly **$+12$ additional LGG patients detected** ($0/12 \to 12/12$).

---

## 8. Part 9 & 10 — Thresholding vs Representation vs Calibration

The audit explicitly demarcates these three distinct concepts:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Model Representation / Latent Discrimination                             │
│    • Measured by: ROC-AUC, PR-AUC                                           │
│    • Invariant to decision threshold shifts                                 │
│    • Reflects whether the network separates class feature embeddings        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Probability Calibration                                                  │
│    • Measured by: Expected Calibration Error (ECE), Brier Score             │
│    • Reflects whether predicted P(y=1) matches empirical accuracy           │
│    • Requires scaling transforms (e.g. Platt scaling, isotonic regression)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Classification Operating-Point Selection (Threshold Tuning)              │
│    • Measured by: Sensitivity (LGG Recall), Specificity (HGG Recall)        │
│    • Selects a decision boundary tau on the existing probability output     │
│    • Shifts position along the invariant ROC curve                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Evaluation of Core Scientific Statement**:
> *"The model contains useful class-separating information, but the default decision threshold ($\tau=0.50$) is poorly aligned with minority-class sensitivity under severe class imbalance."*  
$\implies$ **STRONGLY SUPPORTED INTERPRETATION**. (High ROC-AUC $\ge 0.86$ proves strong latent separation; threshold tuning recovers sensitivity by adjusting the cutoff).

---

## 9. Part 11 & 12 — Post-Hoc Status & Historical $0 \to 0.75$ Discrepancy

1. **Temporal Classification**: Phase 7 threshold optimization was conducted on Day 6 as an **exploratory operating-point sensitivity analysis**. It must be reported as a secondary exploratory finding, NOT as a primary confirmatory result.
2. **Historical ResNet18 Run Separation**: Both historical Run 1 ($0.00$ recall) and canonical Run 3 ($0.75$ recall) were evaluated at default $\tau = 0.50$. The historical difference was driven by **different training epochs and checkpoint states** (Epoch 1 vs Epoch 19), not by post-hoc threshold manipulation.

---

## 10. Part 13 — Candidate Claim Classifications

| Claim ID | Candidate Claim | Classification | Evidence & Justification |
| :---: | :--- | :---: | :--- |
| **Claim A** | *"Threshold optimization improved LGG Recall."* | **SAFE TO CLAIM** | Directly demonstrated across multiple models (e.g., SwinUNETR Prot A $0\% \to 100\%$; ResNet18 Prot A $75\% \to 91.7\%$). |
| **Claim B** | *"Threshold optimization improved minority sensitivity while reducing HGG specificity."* | **SAFE TO CLAIM** | Exact description of operating point trade-off along the ROC curve. |
| **Claim C** | *"The default threshold of 0.50 is suboptimal for all models."* | **CLAIM WITH CAUTION** | Suboptimal for minority sensitivity, but optimal for overall accuracy under 95% majority prevalence. |
| **Claim D** | *"Threshold optimization solves the minority-class problem."* | **NOT SUPPORTED** | Overstated; ROC-AUC is unchanged and majority false positives increase. |
| **Claim E** | *"Some models contain useful discriminative information not reflected at default 0.50."* | **SAFE TO CLAIM** | Supported by high ROC-AUC ($\ge 0.86$) coexisting with 0% sensitivity at $\tau = 0.50$. |
| **Claim F** | *"Poor LGG Recall is purely a threshold problem."* | **NOT SUPPORTED** | Both training loss dynamics and decision thresholds govern end-to-end performance. |
| **Claim G** | *"Threshold optimization demonstrates better clinical performance."* | **NOT SUPPORTED** | Clinical utility depends on cost-benefit trade-offs; threshold adjustment shifts operating points. |
| **Claim H** | *"Threshold selection should be considered an operating-point sensitivity analysis rather than proof of improved representation."* | **SAFE TO CLAIM** | Methodologically exact scientific formulation. |

---

## 11. Part 14 & 15 — Reporting Guidance & Final Forensic Verdict

### 11.1 Recommended Reporting Structure for Publication
- **Primary Result**: Frozen canonical checkpoints evaluated at fixed default $\tau = 0.50$ on the locked test set.
- **Secondary / Exploratory Analysis**: Validation-derived operating-point sensitivity analysis demonstrating latent capacity recovery.

### 11.2 Direct Answers to Audit Questions (Q1 – Q12)
- **Q1. Was threshold selection performed without using locked-test labels?** $\to$ **YES (VERIFIED).**
- **Q2. Was threshold selected from validation data?** $\to$ **YES (VERIFIED).**
- **Q3. Can the optimized threshold be evaluated legitimately on locked test?** $\to$ **YES (VERIFIED), as an out-of-sample exploratory evaluation.**
- **Q4. Is there evidence of test-set leakage during threshold selection?** $\to$ **NO (VERIFIED ABSENT).**
- **Q5. What does the threshold analysis demonstrate?** $\to$ Demonstrates that models have latent discriminative capacity (ROC-AUC $\ge 0.86$) accessible at calibrated operating points.
- **Q6. Does it demonstrate improved LGG sensitivity?** $\to$ **YES.**
- **Q7. Does it demonstrate an LGG sensitivity/specificity trade-off?** $\to$ **YES.**
- **Q8. Does it demonstrate improved representation?** $\to$ **NO (Representation is invariant).**
- **Q9. Does it demonstrate improved calibration?** $\to$ **NO (Calibration requires probability scaling).**
- **Q10. Does it prove the model is clinically better?** $\to$ **NO (Explores trade-offs).**
- **Q11. How should it be reported relative to default $\tau = 0.50$?** $\to$ As a secondary exploratory sensitivity analysis.
- **Q12. What remains unresolved?** $\to$ The multi-fold stability of validation-derived $\tau^*$ across 5-fold cross-validation.
