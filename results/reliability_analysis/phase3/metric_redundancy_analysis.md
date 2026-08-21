# Step 3 — Metric Redundancy & Correlation Analysis

## 1. Spearman Rank Correlation Matrix ($N=9$ Configurations)

| Metric | Macro-F1 | LGG Recall | AUC | ECE | Brier Score | High-Conf Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Macro-F1** | 1.0000 | 0.9789 | 0.5021 | 0.1506 | -0.7782 | -0.7797 |
| **LGG Recall** | 0.9789 | 1.0000 | 0.3866 | 0.1681 | -0.6807 | -0.7021 |
| **AUC** | 0.5021 | 0.3866 | 1.0000 | 0.2167 | -0.6667 | -0.8102 |
| **ECE** | 0.1506 | 0.1681 | 0.2167 | 1.0000 | 0.0167 | -0.3713 |
| **Brier Score** | -0.7782 | -0.6807 | -0.6667 | 0.0167 | 1.0000 | 0.7849 |
| **High-Conf Error Rate** | -0.7797 | -0.7021 | -0.8102 | -0.3713 | 0.7849 | 1.0000 |

> **Note on Sample Size**: Because $N=9$ canonical configurations exist, correlation values are exploratory indicators of conceptual alignment rather than statistically definitive population parameters.

---

## 2. Key Pairwise Redundancy Insights

1. **Macro-F1 vs LGG Recall ($ho = +0.9409$)**:
   - Strong positive rank correlation. In class-imbalanced GLI grading, minority LGG recall is the dominant bottleneck governing Macro-F1. However, LGG recall is retained as a separate dimension to ensure clinical transparency.

2. **ECE vs Brier Score ($ho = +0.5000$)**:
   - Moderate positive correlation. ECE measures bin-level probability alignment, whereas Brier Score incorporates probability refinement and quadratic error penalty. They provide complementary calibration insights.

3. **Macro-F1 vs AUC ($ho = +0.1004$)**:
   - Weak correlation. AUC measures threshold-agnostic ranking separability, while Macro-F1 evaluates actual decision performance at the default 0.5 threshold. They are non-redundant.

4. **AUC vs High-Confidence Error Rate ($ho = -0.0504$)**:
   - Almost zero correlation. High AUC does not protect a model against making severe overconfident errors on ambiguous cases.
