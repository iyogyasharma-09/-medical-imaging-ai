# Step 4 & 5 — Reliability Dimensions & Multi-Metric Dashboard Architecture

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
