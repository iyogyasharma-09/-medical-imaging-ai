# Project Reliability Framework: Dual Dashboard & Gated Index

## 1. Architecture Overview

The project establishes a two-tiered reliability evaluation framework:

### Tier 1 (Primary Ground Truth): Multi-Dimensional Reliability Profile / Dashboard
Independently reports 6 core dimensions:
1. **Test Macro-F1** (Balanced threshold discrimination)
2. **Test LGG Recall** (Minority class sensitivity)
3. **Test AUC** (Threshold-agnostic ranking)
4. **Expected Calibration Error (ECE)** (Probability calibration)
5. **Brier Score** (Probabilistic error refinement)
6. **High-Confidence Error Rate** ($	ext{Conf} \ge 0.80$ overconfidence safety)

### Tier 2 (Secondary Ranking Index): Dimension-Balanced Reliability Score ($R_B$)
Equal 33.3% weighting across 3 functional pillars:
$$R_B = \frac{1}{3} \left( \frac{\text{F1} + \text{LGGRec} + \text{AUC}}{3} \right) + \frac{1}{3} \left( \frac{(1 - \text{ECE}) + (1 - \text{Brier})}{2} \right) + \frac{1}{3} (1 - \text{HighConfErrRate})$$

---

## 2. Mandatory Clinical Safety Gate
To prevent scalar composite scores from hiding total minority class failure:
- **Rule**: If $\text{LGG Recall} < 0.50$, the configuration is marked as **`NON-VIABLE (Gated)`**, regardless of its composite index value.
