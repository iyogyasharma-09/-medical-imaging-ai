# Day 7 — Audit 5: Cross-Validation Scope, Reliability & Generalizability Audit

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Audit Date**: 2026-08-19  
**Audit Mode**: Strict Read-Only Forensic & Statistical Reliability Audit  
**Auditor**: DeepMind Antigravity Advanced Agentic AI  
**Scope**: Phase 6 Five-Fold Cross-Validation on ResNet18 MedicalNet Protocol A ($N=1,063$)  

---

## 1. Executive Summary & Central Question

### 1.1 Purpose of Audit
Phase 6 cross-validation was conducted exclusively on **ResNet18 MedicalNet under Protocol A (unweighted)** across the $N=1,063$ development cohort to evaluate model stability across patient partitions.

The central question addressed by **Audit 5** is:
> *"How reliable is the ResNet18 Protocol A result across different patient partitions, and exactly what can and cannot be generalized from this cross-validation experiment?"*

### 1.2 Central Scientific Finding
1. **Latent Discriminative Representation is Stable**: ResNet18 MedicalNet learns robust, highly separable representations across all five folds ($\text{Mean ROC-AUC} = \mathbf{0.9163 \pm 0.0446}$, range: $0.8629\text{--}0.9634$; $\text{Coefficient of Variation} = \mathbf{4.87\%}$).
2. **Fixed-Threshold Minority Sensitivity is Extremely Unstable**: At the default classification threshold ($\tau = 0.50$), LGG Recall swings from **0.00% to 80.00%** ($\text{Mean LGG Recall} = \mathbf{0.5055 \pm 0.3274}$; $\text{Coefficient of Variation} = \mathbf{64.77\%}$).
3. **Generalizability Boundary**: These empirical variance findings apply strictly to ResNet18 Protocol A on this dataset and **must not be generalized** to DenseNet121, SwinUNETR, Protocol B, or Protocol C without dedicated cross-validation runs.

---

## 2. Part 1 & 2 — CV Design & Data Integrity Verification

The cross-validation split manifest (`cv_split_manifest.json`) and preflight audit records were verified:

### 2.1 Design Parameters
- **Cohort**: Full BraTS 2023 GLI Development Cohort ($N = 1,063$).
- **Class Composition**: 53 LGG (4.99%), 1,010 HGG (95.01%).
- **Number of Folds**: $K = 5$.
- **Partitioning Method**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Subject-Level Isolation**: Splitting performed strictly on unique subject IDs (zero slice/scan leakage).
- **Nominal Train/Validation Ratio**: $\approx 80\% / 20\%$ per fold.
- **Coverage**: All 1,063 unique development subjects appear in validation **exactly once** across the 5 folds ($\text{Coverage} = \mathbf{100\%}$).
- **Train/Validation Overlap**: $\mathbf{0}$ across all 5 folds (**PROVEN ABSENT**).
- **Locked-Test Overlap**: $\mathbf{0}$ across all 5 folds (**PROVEN ABSENT**).

### 2.2 Fold Composition Matrix

| Fold | Train Total | Train LGG | Train HGG | Train LGG % | Val Total | Val LGG | Val HGG | Val LGG % | Train $\cap$ Val | Val $\cap$ Test |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 850 | 42 | 808 | 4.94% | 213 | 11 | 202 | 5.16% | 0 | 0 |
| **Fold 2** | 850 | 42 | 808 | 4.94% | 213 | 11 | 202 | 5.16% | 0 | 0 |
| **Fold 3** | 850 | 42 | 808 | 4.94% | 213 | 11 | 202 | 5.16% | 0 | 0 |
| **Fold 4** | 851 | 43 | 808 | 5.05% | 212 | 10 | 202 | 4.72% | 0 | 0 |
| **Fold 5** | 851 | 43 | 808 | 5.05% | 212 | 10 | 202 | 4.72% | 0 | 0 |
| **Total Unique** | **1,063** | **53** | **1,010** | **4.99%** | **1,063** | **53** | **1,010** | **4.99%** | **0** | **0** |

---

## 3. Part 3 & 4 — Metric Reproduction & Statistical Variability Analysis

All fold metrics were independently recalculated and verified against saved fold training histories and checkpoint state dictionaries:

### 3.1 Fold-by-Fold Performance Breakdown

| Fold | Total Ep | Sel Ep | Checkpoint SHA-256 (Prefix) | Accuracy | Macro-F1 | LGG Recall | HGG Recall | ROC-AUC | Val Loss | Val LGG (Total / Correct) | Case Granularity |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 13 | 8 | `b1f75ee8...` | 0.9531 | 0.7952 | **0.7273** | 0.9653 | 0.9185 | 0.1764 | 11 / 8 | 9.09 pp / case |
| **Fold 2** | 9 | 4 | `96588624...` | 0.9437 | 0.7542 | **0.6364** | 0.9604 | 0.9563 | 0.1090 | 11 / 7 | 9.09 pp / case |
| **Fold 3** | 10 | 5 | `756d3728...` | 0.9624 | 0.7402 | **0.3636** | 0.9950 | 0.8803 | 0.1650 | 11 / 4 | 9.09 pp / case |
| **Fold 4** | 6 | 1 | `83484551...` | 0.9528 | 0.4879 | **0.0000** | 1.0000 | 0.8629 | 0.2920 | 10 / 0 | 10.00 pp / case |
| **Fold 5** | 9 | 4 | `51ca38ac...` | 0.9528 | 0.7951 | **0.8000** | 0.9604 | 0.9634 | 0.1210 | 10 / 8 | 10.00 pp / case |

### 3.2 Quantitative Variability Comparison

| Metric | Mean | Standard Deviation (SD) | Minimum | Maximum | Range | Coefficient of Variation (CV %) | Stability Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Overall Accuracy** | **0.9530** | **$\pm 0.0066$** | 0.9437 | 0.9624 | 0.0187 | **0.69%** | **Highly Stable** (Dominated by HGG) |
| **HGG Recall (Specificity)** | **0.9762** | **$\pm 0.0196$** | 0.9604 | 1.0000 | 0.0396 | **2.01%** | **Stable** |
| **ROC-AUC (Discrimination)** | **0.9163** | **$\pm 0.0446$** | 0.8629 | 0.9634 | 0.1005 | **4.87%** | **Stable** |
| **Macro-F1 Score** | **0.7145** | **$\pm 0.1290$** | 0.4879 | 0.7952 | 0.3073 | **18.06%** | **Moderately Variable** |
| **Validation Loss** | **0.1727** | **$\pm 0.0725$** | 0.1090 | 0.2920 | 0.1830 | **42.00%** | **Extremely Unstable** |
| **LGG Recall (Sensitivity)** | **0.5055** | **$\pm 0.3274$** | **0.0000** | **0.8000** | **0.8000** | **64.77%** | **Extremely Unstable** |

**Contrast**: While Accuracy and ROC-AUC exhibit tight clustering ($\text{CV} < 5\%$), minority-class LGG Recall exhibits extreme instability ($\text{CV} = 64.77\%$), swinging across the entire spectrum from complete detection failure ($0.0\%$) to strong sensitivity ($80.0\%$).

---

## 4. Part 5 — Minority-Class Sample-Size & Resolution Effect

The validation partitions contain only 10 to 11 LGG subjects per fold:
- **Folds 1, 2, 3**: 11 LGG subjects $\to$ Each individual patient shifts sensitivity by **$9.09\text{ percentage points}$**.
- **Folds 4, 5**: 10 LGG subjects $\to$ Each individual patient shifts sensitivity by **$10.00\text{ percentage points}$**.

### Wilson Score 95% Confidence Intervals:
- **Fold 1 ($8/11 = 72.73\%$)**: $95\%\text{ CI} = [43.4\%, 90.3\%]$
- **Fold 2 ($7/11 = 63.64\%$)**: $95\%\text{ CI} = [35.4\%, 84.8\%]$
- **Fold 3 ($4/11 = 36.36\%$)**: $95\%\text{ CI} = [15.2\%, 64.6\%]$
- **Fold 4 ($0/10 = 0.00\%$)**: $95\%\text{ CI} = [0.0\%, 27.8\%]$
- **Fold 5 ($8/10 = 80.00\%$)**: $95\%\text{ CI} = [49.0\%, 94.3\%]$

**Scientific Interpretation**: Small sample size creates coarse metric quantization and wide confidence intervals. However, the 8-patient gap between Fold 4 ($0/10$) and Fold 5 ($8/10$) reflects an authentic divergence in model decision thresholds on those patient subsets rather than mere measurement noise.

---

## 5. Part 6 — The "Favorable Split" Question (Locked Test vs CV)

| Evaluation Setting | Cohort Size ($N$) | LGG Cases ($N$) | Accuracy | Macro-F1 | LGG Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Locked-Test Evaluation** | 188 | 12 | **0.9734** | **0.8842** | **0.7500** (9/12) | **0.9830** |
| **5-Fold Cross-Validation** | 1,063 | 53 | **$0.9530 \pm 0.0066$** | **$0.7145 \pm 0.1290$** | **$0.5055 \pm 0.3274$** | **$0.9163 \pm 0.0446$** |

### Critical Scientific Verdict:
1. **Inside Observed Range**: The locked-test LGG Recall ($0.7500$) lies within the empirical range observed during cross-validation ($[0.0000, 0.8000]$).
2. **Upper-Tail Position**: The locked-test result is $+0.2445$ (+24.45 pp) above the cross-validation mean ($50.55\%$), sitting near Fold 5 ($80.0\%$).
3. **Absence of Proven Bias**: The locked test was isolated before training with zero subject overlap. The higher score does **not** prove data snooping or bias; rather, it indicates that single-split evaluations are subject to sampling variance.
4. **Reporting Requirement**: The locked-test result must **not** be presented as a guaranteed average sensitivity, but as a single-split evaluation reported alongside the $50.55\% \pm 32.74\%$ CV distribution.

---

## 6. Part 7 & 8 — Detailed Forensic Analysis of Fold 4 vs Fold 5

Both Fold 4 and Fold 5 contain exactly 10 LGG and 202 HGG validation subjects (212 total), yet produced diametrically opposed sensitivity:

### 6.1 The Mechanism of Fold 4 Collapse ($0.00\%$ Recall)
- **Epoch 1**: Model predicted all 212 validation cases as HGG at threshold $\tau = 0.50$ ($\text{Val LGG Recall} = 0.0000, \text{Val Macro-F1} = 0.4879$). However, continuous separation was already strong ($\text{Val ROC-AUC} = 0.8629$).
- **Epochs 2–6**: Training continued. Training loss decreased from $0.2543 \to 0.0463$. Validation loss reached its minimum of $0.1218$ at Epoch 6, and validation ROC-AUC reached $0.9287$ at Epoch 6.
- **Early Stopping Trigger**: Because validation Macro-F1 at threshold 0.5 remained at $0.4879$ (posterior probabilities for the 10 LGG cases clustered between $0.20\text{--}0.48$, never crossing $0.50$), `EarlyStopping(patience=5)` did not detect an improvement over Epoch 1.
- **Restored State**: Training stopped at Epoch 6, and the checkpoint restored weights from **Epoch 1**, freezing the model in a zero-recall state.
- **Computational Audit**: System logs show no memory errors, corrupted tensors, or worker crashes during Fold 4. The result is a legitimate mathematical consequence of unweighted loss dynamics interacting with patience-based early stopping.

### 6.2 Fold 5 Success ($80.00\%$ Recall)
- In Fold 5, the model surpassed the majority plateau at Epoch 4 ($\text{Val Macro-F1} = 0.7951, \text{Val LGG Recall} = 0.8000, \text{Val ROC-AUC} = 0.9634$), and restored optimal Epoch 4 weights upon early stopping at Epoch 9.

### 6.3 Scientific Takeaway (Fold 4 vs Fold 5)
$\implies$ **ESTABLISHED OBSERVATION**: Under an unweighted cross-entropy loss function and fixed default threshold ($\tau = 0.50$), different patient partitions can yield dramatically different minority-class sensitivity ($0\%$ vs $80\%$) despite sharing identical architecture, hyper-parameters, and top-level random seeds.

---

## 7. Part 9 — Model Quality & Representational Capacity

- **Q1. Does CV support that ResNet18 learns useful discriminative information?**  
  **YES.** Mean ROC-AUC of $0.9163 \pm 0.0446$ across 1,063 development subjects demonstrates strong latent feature separation.
- **Q2. Does high AUC support this?**  
  **YES.** Continuous probabilities rank LGG vs HGG accurately across all 5 folds (minimum AUC is 0.8629 in Fold 4).
- **Q3. Does high Accuracy support this?**  
  **NO.** Accuracy ($95.30\% \pm 0.66\%$) is mathematically dominated by the 95.0% majority class and is uninformative regarding minority sensitivity.
- **Q4. Does high LGG Recall variability mean the model is useless?**  
  **NO.** It means that hard classification decisions at the default $\tau = 0.50$ operating point are uncalibrated and unstable under unweighted loss.
- **Q5. Does CV show minority decision behavior is unstable?**  
  **YES.** $\text{CV} = 64.77\%$ in LGG Recall confirms decision threshold instability.
- **Q6. Does CV establish that the model is robust?**  
  **PARTIALLY.** Robust in latent ranking (ROC-AUC), but non-robust in thresholded sensitivity.
- **Q7. Recommended Precise Formulation**:  
  > *"ResNet18 MedicalNet maintained stable overall discriminative capability (ROC-AUC = $0.9163 \pm 0.0446$) but exhibited substantial fold-to-fold sensitivity variability on the minority class ($0.5055 \pm 0.3274$) under unweighted training and fixed default thresholding."*

---

## 8. Part 10 & 11 — Generalizability Boundary & What CV Does Not Answer

| Finding / Scope | Empirical Evidence | Generalizable to DenseNet121? | Generalizable to SwinUNETR? | Generalizable to Protocols B & C? | Scientific Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ResNet18 Protocol A has high LGG Recall variance ($0.00\text{--}0.80$)** | Phase 6 5-fold CV ($N=1,063$) | ❌ **UNPROVEN** | ❌ **UNPROVEN** | ❌ **UNPROVEN** | **ESTABLISHED FACT (ResNet18 Prot A)** |
| **ROC-AUC is stable across patient folds ($0.916 \pm 0.045$)** | Phase 6 5-fold CV ($N=1,063$) | ⚠️ **PLAUSIBLE HYPOTHESIS** | ⚠️ **PLAUSIBLE HYPOTHESIS** | ⚠️ **PLAUSIBLE HYPOTHESIS** | **ESTABLISHED FACT (ResNet18 Prot A)** |
| **Accuracy is uninformative under 19:1 imbalance ($\approx 95\%$)** | Mathematical prevalence property | ✅ **SAFE TO GENERALIZE** | ✅ **SAFE TO GENERALIZE** | ✅ **SAFE TO GENERALIZE** | **ESTABLISHED MATHEMATICAL FACT** |
| **Focal Loss will eliminate sensitivity variance across folds** | Untested hypothesis | ❌ **UNTESTED HYPOTHESIS** | ❌ **UNTESTED HYPOTHESIS** | ❌ **UNTESTED HYPOTHESIS** | **RESEARCH HYPOTHESIS** |

### Questions That Phase 6 Cannot Answer:
1. The exact quantitative partition of variance between **patient sampling** vs **multi-seed training stochasticity** on a fixed split.
2. Whether DenseNet121 or SwinUNETR exhibits similar fold-to-fold sensitivity variance under 5-fold cross-validation.
3. Whether Protocol B (inverse frequency) or Protocol C (minority-boosted) stabilizes or exacerbates inter-fold variance.
4. Whether Focal Loss provably reduces sensitivity variance across 5 folds.

---

## 9. Part 12 & 13 — Candidate Claim Classifications

| Claim ID | Candidate Claim | Classification | Evidence & Justification |
| :---: | :--- | :---: | :--- |
| **Claim A** | *"ResNet18 achieved 95.3% accuracy."* | **CLAIM WITH CAUTION** | Factual mean, but conceals $0\text{--}80\%$ minority sensitivity swings. |
| **Claim B** | *"ResNet18 achieved 95.3 ± 0.7% accuracy across five folds."* | **SAFE TO CLAIM** | Exact statistical reporting of cross-validation accuracy. |
| **Claim C** | *"ResNet18 achieved 50.6 ± 32.7% LGG Recall across five folds."* | **SAFE TO CLAIM** | Exact statistical reporting of cross-validation minority sensitivity. |
| **Claim D** | *"ResNet18 was unstable."* | **CLAIM WITH CAUTION** | Overly broad; discrimination was stable ($\text{CV}=4.9\%$), but minority thresholding was unstable ($\text{CV}=64.8\%$). |
| **Claim E** | *"ResNet18 maintained high overall discrimination but showed substantial variability in minority-class sensitivity."* | **SAFE TO CLAIM** | Most accurate, nuanced scientific summary of the experimental data. |
| **Claim F** | *"The model's performance depends on patient distribution."* | **SAFE TO CLAIM** | Directly demonstrated by $0\%$ vs $80\%$ recall across identical-pipeline folds. |
| **Claim G** | *"Patient distribution caused the LGG instability."* | **CLAIM WITH CAUTION** | Patient distribution is a major contributor, but optimization dynamics and fixed thresholding also contribute. |
| **Claim H** | *"The locked-test result of 75% LGG Recall is representative of expected performance."* | **NOT SUPPORTED** | Contradicted by CV mean ($50.6\%$); 75% represents an upper-tail outcome. |
| **Claim I** | *"The locked-test result lies within the range observed during cross-validation but above its mean."* | **SAFE TO CLAIM** | Factually precise description ($75.0\% \in [0.0\%, 80.0\%]$ and $> 50.55\%$). |

---

## 10. Part 14 — Final Scientific Verdict (Q1 – Q10)

- **Q1. Is the CV experiment internally valid?**  
  **YES (VERIFIED).** Stratified subject-level splitting, zero data leakage, 100% development coverage.
- **Q2. Is the CV result reproducible from saved artifacts?**  
  **YES (VERIFIED).** Restored model states, history metrics, and checkpoint hashes match 100%.
- **Q3. Does CV reveal hidden minority-class instability?**  
  **YES.** Accuracy remains constant at ~95.3% while LGG Recall swings from 0.0% to 80.0%.
- **Q4. Does CV support that overall discrimination is more stable than LGG sensitivity?**  
  **YES.** $\text{CV} = 4.87\%$ for ROC-AUC vs $\text{CV} = 64.77\%$ for LGG Recall.
- **Q5. Does CV explain the historical 0 $\to$ 0.75 LGG Recall discrepancy?**  
  **PARTIALLY.** Demonstrates that 0.00% recall occurs naturally on unfavorable splits with early stopping (Fold 4), while 75–80% occurs on favorable splits with longer optimization (Fold 5).
- **Q6. Does CV establish patient-distribution effects?**  
  **YES.** Different patient subsets yield divergent sensitivity under identical pipelines.
- **Q7. Does CV establish training stochasticity as the primary cause?**  
  **NO.** Multi-seed fixed-split variance has not been decoupled.
- **Q8. Does CV establish that the 0.75 locked-test result is unusually favorable?**  
  **NO.** It establishes that 0.75 lies in the upper tail of the empirical distribution, but does not prove statistical bias.
- **Q9. What is the strongest scientifically defensible interpretation?**  
  > *"ResNet18 MedicalNet learns stable, separable 3D feature representations (ROC-AUC = $0.9163 \pm 0.0446$), but fixed-threshold minority sensitivity exhibits substantial fold-to-fold variability ($0.5055 \pm 0.3274$) under unweighted cross-entropy loss."*
- **Q10. What important questions remain for future experiments?**  
  1. A 5-seed fixed-split ablation to isolate pure optimization stochasticity.  
  2. 5-fold cross-validation on Protocols B and C and alternative architectures (DenseNet121, SwinUNETR).  
  3. Evaluation of dynamically modulated Focal Loss under 5-fold cross-validation.
