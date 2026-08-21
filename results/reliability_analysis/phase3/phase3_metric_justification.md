# Step 2 — Scientific Justification of Candidate Reliability Metrics

This report evaluates 6 candidate metrics across 8 explicit scientific criteria to determine their role in the GLI 3D grading reliability framework.

---

## 1. Metric-by-Metric Scientific Evaluation

### 1.1 Test Macro-F1
- **A. Measurement**: Unweighted harmonic mean of F1 scores across LGG (class 0) and HGG (class 1).
- **B. Clinical Relevance**: Highly relevant. GLI datasets are class-imbalanced (~93.6% HGG). Standard accuracy obscures minority performance, whereas Macro-F1 weights both grades equally.
- **C. Dimension**: **Discriminative Performance** (Operating threshold = 0.5).
- **D. Direction**: Higher is better ($1.0 = $ perfect).
- **E. Unique Value**: Captures balanced decision quality across both classes at the default decision boundary.
- **F. Redundancy**: Partially overlaps with LGG Recall, but incorporates precision and majority HGG performance.
- **G. Inclusion Recommendation**: **INCLUDE** as the primary discriminative performance metric.
- **H. Exclusion Rationale**: N/A.

---

### 1.2 Test LGG Recall (Specificity / Minority Sensitivity)
- **A. Measurement**: Fraction of true LGG cases correctly identified ($TN / (TN + FP)$).
- **B. Clinical Relevance**: **Critical safety metric**. Misclassifying an LGG patient as HGG leads to overtreatment, aggressive radiation, and unnecessary toxicity.
- **C. Dimension**: **Minority-Class Sensitivity / Clinical Safety**.
- **D. Direction**: Higher is better ($1.0 = $ perfect).
- **E. Unique Value**: Directly measures minority class protection, which is otherwise masked by dominant HGG numbers.
- **F. Redundancy**: Contributes to Macro-F1, but isolates minority risk specifically.
- **G. Inclusion Recommendation**: **INCLUDE** as a standalone mandatory component.
- **H. Exclusion Rationale**: N/A.

---

### 1.3 Test AUC (ROC Area Under Curve)
- **A. Measurement**: Threshold-agnostic probability ranking quality across all cutoffs.
- **B. Clinical Relevance**: High. Evaluates whether predicted probabilities separate LGG from HGG regardless of the 0.5 decision threshold.
- **C. Dimension**: **Discriminative Ranking Capability**.
- **D. Direction**: Higher is better ($1.0 = $ perfect).
- **E. Unique Value**: Independent of decision cutoff, evaluating intrinsic feature separability.
- **F. Redundancy**: Complementary to Macro-F1. SwinUNETR shows high AUC (0.9735) despite poor fixed-threshold Macro-F1 (0.6736).
- **G. Inclusion Recommendation**: **INCLUDE**.
- **H. Exclusion Rationale**: N/A.

---

### 1.4 Expected Calibration Error (ECE)
- **A. Measurement**: Average weighted discrepancy between predicted class confidence and observed accuracy across 10 probability bins.
- **B. Clinical Relevance**: **Critical probabilistic metric**. Clinicians rely on probability estimates for risk communication; overconfident or underconfident probabilities misinform treatment planning.
- **C. Dimension**: **Probabilistic Calibration Reliability**.
- **D. Direction**: Lower is better ($0.0 = $ perfect calibration).
- **E. Unique Value**: Directly measures probability alignment without regard to classification threshold choices.
- **F. Redundancy**: Distinct from discriminative metrics.
- **G. Inclusion Recommendation**: **INCLUDE** (transformed via $1 - ECE$).
- **H. Exclusion Rationale**: N/A.

---

### 1.5 Brier Score
- **A. Measurement**: Mean squared error between predicted probability vectors and one-hot true class vectors.
- **B. Clinical Relevance**: High. Heavily penalizes confident wrong predictions (quadratic loss).
- **C. Dimension**: **Probabilistic Refinement & Calibration**.
- **D. Direction**: Lower is better ($0.0 = $ perfect prediction).
- **E. Unique Value**: Combines calibration and sharpness into a proper scoring rule.
- **F. Redundancy**: Moderately correlated with ECE, but penalizes severe outliers more aggressively.
- **G. Inclusion Recommendation**: **INCLUDE** (transformed via $1 - Brier$).
- **H. Exclusion Rationale**: N/A.

---

### 1.6 High-Confidence Error Rate ($	ext{Conf} \ge 0.80$)
- **A. Measurement**: Proportion of test cohort misclassified with high probability ($\ge 0.80$).
- **B. Clinical Relevance**: **Critical safety metric**. Confidently wrong AI predictions erode clinician trust and cause high clinical harm.
- **C. Dimension**: **Overconfidence Safety Risk**.
- **D. Direction**: Lower is better ($0.0 = $ zero overconfident errors).
- **E. Unique Value**: Specifically targets dangerous false certainty, which ECE averages out over bins.
- **F. Redundancy**: Non-redundant with Macro-F1 or AUC.
- **G. Inclusion Recommendation**: **INCLUDE** (transformed via $1 - Rate$).
- **H. Exclusion Rationale**: N/A.
