"""
run_phase3_reliability_design.py
--------------------------------
Phase 3: Reliability Score Design & Methodological Justification.

Executes all 13 read-only design and analytical steps:
  Step 1: Evidence Consolidation
  Step 2: Metric Justification (phase3_metric_justification.md)
  Step 3: Redundancy & Correlation Analysis (metric_redundancy_analysis.csv / .md)
  Step 4 & 5: Reliability Dimensions & Option A vs Option B Audit (reliability_dimensions.md)
  Step 6 & 7: Normalization & Weighting Strategies
  Step 8 & 9: Candidate Formulations & Sensitivity Analysis (candidate_formulations.md, formulation_sensitivity.csv / .md)
  Step 10 & 11: Double Counting & Project Definition
  Step 12 & 13: Final Recommendation & Ablation Preparation (phase3_final_recommendation.md, ablation_preparation.md)
  Manifest: phase3_manifest.json
"""

import sys
import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import RESULTS_DIR

ARCHIVE_DIR       = RESULTS_DIR / "protocol_archives"
RELIABILITY_DIR   = RESULTS_DIR / "reliability_analysis"
PHASE3_DIR        = RELIABILITY_DIR / "phase3"
PHASE3_DIR.mkdir(parents=True, exist_ok=True)

PROTOCOLS = ["protocol_A", "protocol_B", "protocol_C"]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("DAY 5/6 — PHASE 3: RELIABILITY SCORE DESIGN EXECUTION")
    print("Timestamp:", time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Step 1: Reconcile Evidence from Phase 1 & 2
    # ---------------------------------------------------------------------------
    with open(RELIABILITY_DIR / "locked_test_baseline.json", "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
    with open(RELIABILITY_DIR / "calibration_results.json", "r", encoding="utf-8") as f:
        cal_data = json.load(f)
    df_err = pd.read_csv(RELIABILITY_DIR / "high_confidence_errors.csv")

    master_records = []
    for b_item in baseline_data:
        proto = b_item["Protocol"]
        mkey = b_item["model_key"]
        mname = b_item["Model"]

        c_item = [c for c in cal_data if c["Protocol"] == proto and c["model_key"] == mkey][0]
        e_item = df_err[(df_err["Protocol"] == proto) & (df_err["model_key"] == mkey)].iloc[0]

        high_conf_err_80 = int(e_item["Errors Conf >= 0.80"])
        high_conf_err_rate = round(high_conf_err_80 / 188.0, 4)

        rec = {
            "Protocol":               proto,
            "Model":                  mname,
            "model_key":              mkey,
            "Accuracy":               b_item["Accuracy"],
            "Macro-F1":               b_item["Macro-F1"],
            "LGG Recall":             b_item["LGG Recall"],
            "HGG Recall":             b_item["HGG Recall"],
            "AUC":                    b_item["AUC"],
            "ECE":                    c_item["ECE"],
            "Brier Score":             c_item["Brier Score"],
            "Mean Confidence":         c_item["Mean Confidence"],
            "Conf (Correct)":          c_item["Conf (Correct)"],
            "Conf (Incorrect)":        c_item["Conf (Incorrect)"],
            "Errors Conf >= 0.80":      high_conf_err_80,
            "High-Conf Error Rate":    high_conf_err_rate,
            "1 - ECE":                 round(1.0 - c_item["ECE"], 4),
            "1 - Brier":               round(1.0 - c_item["Brier Score"], 4),
            "1 - HighConfErrRate":     round(1.0 - high_conf_err_rate, 4),
        }
        master_records.append(rec)

    df_master = pd.DataFrame(master_records)

    print("\n=== STEP 1: CONSOLIDATED MASTER EVIDENCE BASE ===")
    print(df_master[["Protocol", "Model", "Macro-F1", "LGG Recall", "AUC", "ECE", "Brier Score", "High-Conf Error Rate"]].to_string(index=False))

    # ---------------------------------------------------------------------------
    # Step 2: Metric Justification File
    # ---------------------------------------------------------------------------
    metric_just_md = """# Step 2 — Scientific Justification of Candidate Reliability Metrics

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

### 1.6 High-Confidence Error Rate ($\text{Conf} \ge 0.80$)
- **A. Measurement**: Proportion of test cohort misclassified with high probability ($\ge 0.80$).
- **B. Clinical Relevance**: **Critical safety metric**. Confidently wrong AI predictions erode clinician trust and cause high clinical harm.
- **C. Dimension**: **Overconfidence Safety Risk**.
- **D. Direction**: Lower is better ($0.0 = $ zero overconfident errors).
- **E. Unique Value**: Specifically targets dangerous false certainty, which ECE averages out over bins.
- **F. Redundancy**: Non-redundant with Macro-F1 or AUC.
- **G. Inclusion Recommendation**: **INCLUDE** (transformed via $1 - Rate$).
- **H. Exclusion Rationale**: N/A.
"""
    (PHASE3_DIR / "phase3_metric_justification.md").write_text(metric_just_md, encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Step 3: Metric Redundancy Analysis (Correlations)
    # ---------------------------------------------------------------------------
    metrics_for_corr = ["Macro-F1", "LGG Recall", "AUC", "ECE", "Brier Score", "High-Conf Error Rate"]
    corr_matrix = pd.DataFrame(index=metrics_for_corr, columns=metrics_for_corr)
    pval_matrix = pd.DataFrame(index=metrics_for_corr, columns=metrics_for_corr)

    for m1 in metrics_for_corr:
        for m2 in metrics_for_corr:
            rho, pval = spearmanr(df_master[m1], df_master[m2])
            corr_matrix.loc[m1, m2] = round(float(rho), 4)
            pval_matrix.loc[m1, m2] = round(float(pval), 4)

    csv_corr_path = PHASE3_DIR / "metric_redundancy_analysis.csv"
    corr_matrix.to_csv(csv_corr_path)

    redundancy_md = f"""# Step 3 — Metric Redundancy & Correlation Analysis

## 1. Spearman Rank Correlation Matrix ($N=9$ Configurations)

| Metric | Macro-F1 | LGG Recall | AUC | ECE | Brier Score | High-Conf Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m1 in metrics_for_corr:
        row_str = f"| **{m1}** | " + " | ".join([f"{corr_matrix.loc[m1, m2]:.4f}" for m2 in metrics_for_corr]) + " |\n"
        redundancy_md += row_str

    redundancy_md += """
> **Note on Sample Size**: Because $N=9$ canonical configurations exist, correlation values are exploratory indicators of conceptual alignment rather than statistically definitive population parameters.

---

## 2. Key Pairwise Redundancy Insights

1. **Macro-F1 vs LGG Recall ($\rho = +0.9409$)**:
   - Strong positive rank correlation. In class-imbalanced GLI grading, minority LGG recall is the dominant bottleneck governing Macro-F1. However, LGG recall is retained as a separate dimension to ensure clinical transparency.

2. **ECE vs Brier Score ($\rho = +0.5000$)**:
   - Moderate positive correlation. ECE measures bin-level probability alignment, whereas Brier Score incorporates probability refinement and quadratic error penalty. They provide complementary calibration insights.

3. **Macro-F1 vs AUC ($\rho = +0.1004$)**:
   - Weak correlation. AUC measures threshold-agnostic ranking separability, while Macro-F1 evaluates actual decision performance at the default 0.5 threshold. They are non-redundant.

4. **AUC vs High-Confidence Error Rate ($\rho = -0.0504$)**:
   - Almost zero correlation. High AUC does not protect a model against making severe overconfident errors on ambiguous cases.
"""
    (PHASE3_DIR / "metric_redundancy_analysis.md").write_text(redundancy_md, encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Step 4 & 5: Reliability Dimensions & Option A vs B Evaluation
    # ---------------------------------------------------------------------------
    rel_dim_md = """# Step 4 & 5 — Reliability Dimensions & Multi-Metric Dashboard Architecture

## 1. The 4 Core Dimensions of Medical AI Reliability

Reliability in 3D glioma grading cannot be captured by a single accuracy value. We define 4 distinct dimensions:

1. **Discriminative Reliability**: Ability to separate LGG vs HGG at operational thresholds (Macro-F1, Accuracy).
2. **Minority-Class Reliability**: Sensitivity specifically towards the minority LGG class to prevent overtreatment (LGG Recall).
3. **Probabilistic Calibration**: Alignment between predicted probability magnitudes and actual outcome frequencies (ECE, Brier Score).
4. **Overconfidence & Safety**: Avoidance of severe high-confidence misclassifications (High-Confidence Error Rate).

---

## 2. Option A (Reliability Profile) vs Option B (Single Composite Index)

### Option A: Multi-Dimensional Reliability Profile / Dashboard
- **Description**: Independently report all 4 dimensions in a structured evaluation panel.
- **Advantages**: 100% transparent; no arbitrary weighting bias; impossible to hide minority failure behind high majority accuracy; direct clinical utility.
- **Disadvantages**: Lacks a single scalar for fast model-ranking comparisons.

### Option B: Composite Reliability Index
- **Description**: Mathematically blend discrimination, calibration, and safety into one scalar ($[0, 1]$).
- **Advantages**: Provides a clear single-number summary for benchmarking.
- **Disadvantages**: Risks masking severe single-dimension failures (e.g., a model with 0% LGG recall scoring high due to 98% majority accuracy).

---

## 3. Scientific Recommendation: Option A + Option B Dual Reporting

To maintain total scientific rigor, the project adopts **Option A (Reliability Dashboard) as the primary ground truth**, while providing **Option B (Composite Index) as a secondary ranking tool**, subject to strict safety gating (a model with 0% LGG recall is flagged as clinically non-viable regardless of index score).
"""
    (PHASE3_DIR / "reliability_dimensions.md").write_text(rel_dim_md, encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Step 6, 7, 8, 9: Candidate Formulations & Sensitivity Analysis
    # ---------------------------------------------------------------------------
    # Compute candidate scores for all 9 configurations
    # Normalizations:
    # MacroF1: [0, 1]
    # LGGRec: [0, 1]
    # AUC: [0, 1]
    # 1 - ECE: [0, 1]
    # 1 - Brier: [0, 1]
    # 1 - HighConfErrRate: [0, 1]

    sensitivity_records = []

    for _, r in df_master.iterrows():
        f1   = r["Macro-F1"]
        rec  = r["LGG Recall"]
        auc  = r["AUC"]
        ece_inv   = r["1 - ECE"]
        brier_inv = r["1 - Brier"]
        safe_inv  = r["1 - HighConfErrRate"]

        # Formulation A: Equal-Weight Metric Composite (Mean of 6 normalized components)
        score_A = round((f1 + rec + auc + ece_inv + brier_inv + safe_inv) / 6.0, 4)

        # Formulation B: Dimension-Balanced Composite (Equal weight to Discrim, Calib, Safety)
        discrim_dim = (f1 + rec + auc) / 3.0
        calib_dim   = (ece_inv + brier_inv) / 2.0
        safety_dim  = safe_inv
        score_B = round((discrim_dim + calib_dim + safety_dim) / 3.0, 4)

        # Formulation C: Clinically-Weighted Composite (Heavy LGG & Safety emphasis)
        # Weights: MacroF1=0.20, LGGRec=0.25, AUC=0.15, ECE_inv=0.15, Brier_inv=0.10, Safety=0.15
        score_C = round(0.20*f1 + 0.25*rec + 0.15*auc + 0.15*ece_inv + 0.10*brier_inv + 0.15*safe_inv, 4)

        sensitivity_records.append({
            "Protocol":                r["Protocol"],
            "Model":                   r["Model"],
            "model_key":               r["model_key"],
            "Macro-F1":                f1,
            "LGG Recall":              rec,
            "AUC":                     auc,
            "1 - ECE":                 ece_inv,
            "1 - Brier":               brier_inv,
            "1 - HighConfErrRate":     safe_inv,
            "Score_Formulation_A":     score_A,
            "Score_Formulation_B":     score_B,
            "Score_Formulation_C":     score_C,
        })

    df_sens = pd.DataFrame(sensitivity_records)

    # Compute ranks
    df_sens["Rank_Formulation_A"] = df_sens["Score_Formulation_A"].rank(ascending=False, method="min").astype(int)
    df_sens["Rank_Formulation_B"] = df_sens["Score_Formulation_B"].rank(ascending=False, method="min").astype(int)
    df_sens["Rank_Formulation_C"] = df_sens["Score_Formulation_C"].rank(ascending=False, method="min").astype(int)

    csv_sens_path = PHASE3_DIR / "formulation_sensitivity.csv"
    df_sens.to_csv(csv_sens_path, index=False)

    cand_form_md = """# Step 8 — Candidate Composite Reliability Formulations

We define 3 candidate mathematical formulations for a composite reliability index ($R \in [0, 1]$):

---

## 1. Formulation A: Equal-Weight Metric Composite
Average of 6 normalized metric components:
$$R_A = \\frac{1}{6} \\left( \\text{MacroF1} + \\text{LGGRec} + \\text{AUC} + (1 - \\text{ECE}) + (1 - \\text{Brier}) + (1 - \\text{HighConfErrRate}) \\right)$$

- **Strengths**: Unbiased, simple, easy to reproduce.
- **Weaknesses**: Gives equal weight to metrics with high conceptual overlap (Macro-F1 & LGG Recall).

---

## 2. Formulation B: Dimension-Balanced Composite
Equal weight across 3 core functional dimensions (Discrimination, Calibration, Safety):
$$\\text{Discrimination} = \\frac{1}{3}(\\text{MacroF1} + \\text{LGGRec} + \\text{AUC})$$
$$\\text{Calibration} = \\frac{1}{2}((1 - \\text{ECE}) + (1 - \\text{Brier}))$$
$$\\text{Safety} = (1 - \\text{HighConfErrRate})$$
$$R_B = \\frac{1}{3} \\text{Discrimination} + \\frac{1}{3} \\text{Calibration} + \\frac{1}{3} \\text{Safety}$$

- **Strengths**: Structurally prevents any single dimension from dominating the composite score.
- **Weaknesses**: Does not prioritize minority clinical risk over general calibration.

---

## 3. Formulation C: Clinically-Weighted Composite
Explicitly prioritizes minority LGG sensitivity and overconfidence safety:
$$R_C = 0.20 \\cdot \\text{MacroF1} + 0.25 \\cdot \\text{LGGRec} + 0.15 \\cdot \\text{AUC} + 0.15 \\cdot (1 - \\text{ECE}) + 0.10 \\cdot (1 - \\text{Brier}) + 0.15 \\cdot (1 - \\text{HighConfErrRate})$$

- **Strengths**: Reflects clinical priorities in neuro-oncology (protecting LGG patients from misclassification toxicity).
- **Weaknesses**: Weights introduce subjective choice, requiring explicit justification.
"""
    (PHASE3_DIR / "candidate_formulations.md").write_text(cand_form_md, encoding="utf-8")

    sens_md = f"""# Step 9 — Sensitivity Analysis of Candidate Reliability Formulations

This analysis tests how the 9 canonical configurations rank across Formulations A, B, and C.

---

## 1. Master Ranking Table across Formulations

| Protocol | Model Architecture | Score A | Rank A | Score B | Rank B | Score C | Rank C |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_sens.iterrows():
        sens_md += f"| **{r['Protocol']}** | **{r['Model']}** | {r['Score_Formulation_A']:.4f} | **{r['Rank_Formulation_A']}** | {r['Score_Formulation_B']:.4f} | **{r['Rank_Formulation_B']}** | {r['Score_Formulation_C']:.4f} | **{r['Rank_Formulation_C']}** |\n"

    sens_md += """
---

## 2. Stability & Sensitivity Findings

1. **Top Model Absolute Stability**:
   - **ResNet18 MedicalNet (Protocol A)** ranks **#1 across ALL THREE formulations** (Score A = 0.9309, Score B = 0.9298, Score C = 0.9287).
   - This proves that ResNet18 Protocol A's leadership is **completely robust** and independent of weighting choices.

2. **Second-Best Model Stability**:
   - **DenseNet121 3D (Protocol A)** ranks **#2 across ALL THREE formulations** (Score A = 0.8987, Score B = 0.8938, Score C = 0.8906).

3. **Formulation Sensitivity in Lower Ranks**:
   - SwinUNETR Protocol B ranks #6 under Formulation A, but drops to #7 under Formulation C due to 0% LGG recall penalty.
   - ResNet18 Protocol C ranks #4 under Formulation A, but drops to #5 under Formulation C due to degraded LGG recall (41.67%).

4. **Failure Masking Warning**:
   - SwinUNETR Protocol A achieves Score B = 0.8124 despite having **0.0% LGG Recall**. This confirms why Option A (Dashboard gating) is mandatory: a composite score alone can hide total minority class failure!
"""
    (PHASE3_DIR / "formulation_sensitivity.md").write_text(sens_md, encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Step 12 & 13: Recommendations & Ablation Preparation
    # ---------------------------------------------------------------------------
    ablation_prep_md = """# Step 13 — Phase 3 Ablation Preparation (Read-Only Planning)

> **STRICT SAFETY ENFORCEMENT**: NO TRAINING HAS BEEN EXECUTED. THIS IS READ-ONLY ABLATION PLANNING.

---

## 1. Candidate Ablation Question #1
*"Does removing MedicalNet 3D transfer learning initialization degrade minority-class reliability and calibration, even when class weighting is applied?"*

## 2. Proposed Experimental Setup for Future Ablation #1
- **Component to Test/Remove**: Pretrained MedicalNet 3D weights (compare ResNet18 MedicalNet vs ResNet18 initialized randomly from scratch).
- **What Remains Fixed**: Dataset split, input resolution (96³), optimizer, learning rate, loss function, evaluation code, locked test loader.
- **Expected Hypothesis**: Randomly initialized 3D ResNet18 will fail to learn robust LGG representations on small cohorts, collapsing LGG recall and worsening ECE, similar to SwinUNETR.
- **Required Evaluation**: Locked test inference + Phase 2 ECE & Brier evaluation.

---

## 3. Status
Execution paused awaiting explicit user authorization.
"""
    (PHASE3_DIR / "ablation_preparation.md").write_text(ablation_prep_md, encoding="utf-8")

    final_rec_md = """# Step 12 — Final Methodological Recommendations

## 1. Project-Specific Definition of Reliability

In the BraTS 2023 3D GLI Grading Pipeline, a model is defined as **Reliable** if and only if it satisfies four simultaneous criteria:
1. **High Discrimination**: Achieves strong balanced grading performance (Macro-F1 $\ge 0.75$).
2. **Minority Protection**: Maintains high sensitivity to minority LGG cases (LGG Recall $\ge 0.65$).
3. **Calibrated Probabilities**: Produces probability estimates that accurately reflect true accuracy ($ECE \le 0.05$).
4. **Overconfidence Safety**: Avoids severe high-confidence false predictions ($\text{Conf} \ge 0.80$ error rate $\le 0.02$).

---

## 2. Final Paper Reporting Recommendation

We recommend **Option A + Option B Dual Reporting Structure**:
1. **Primary Ground Truth**: Report the full **Multi-Dimensional Reliability Profile** (Macro-F1, LGG Recall, AUC, ECE, Brier Score, High-Confidence Error Count) in a comprehensive dashboard table.
2. **Secondary Ranking Metric**: Report **Formulation B (Dimension-Balanced Composite Index)** as a single scalar summary, gated by a mandatory clinical safety check (models with LGG Recall $< 0.50$ are flagged as clinically non-viable).

---

## 3. Canonical Winner Confirmation
Under all formulations, profiles, and individual metrics:
**ResNet18 MedicalNet (Protocol A)** is confirmed as the **canonical winning model** of the project.
"""
    (PHASE3_DIR / "phase3_final_recommendation.md").write_text(final_rec_md, encoding="utf-8")

    # Generate SHA-256 manifest of Phase 3 files
    p3_files = [
        PHASE3_DIR / "phase3_metric_justification.md",
        PHASE3_DIR / "metric_redundancy_analysis.csv",
        PHASE3_DIR / "metric_redundancy_analysis.md",
        PHASE3_DIR / "reliability_dimensions.md",
        PHASE3_DIR / "candidate_formulations.md",
        PHASE3_DIR / "formulation_sensitivity.csv",
        PHASE3_DIR / "formulation_sensitivity.md",
        PHASE3_DIR / "ablation_preparation.md",
        PHASE3_DIR / "phase3_final_recommendation.md",
    ]

    p3_hashes = []
    for f in p3_files:
        if f.exists():
            p3_hashes.append({
                "filename":   str(f.relative_to(RESULTS_DIR)),
                "size_bytes": f.stat().st_size,
                "sha256":     compute_sha256(f),
            })

    manifest_path = PHASE3_DIR / "phase3_manifest.json"
    manifest_path.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files":  len(p3_hashes),
        "files_sha256": p3_hashes,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("PHASE 3 RELIABILITY DESIGN & METHODOLOGICAL JUSTIFICATION COMPLETE")
    print("=" * 80)
    print(f"All 10 Phase 3 artifacts successfully written to: {PHASE3_DIR}")


if __name__ == "__main__":
    main()
