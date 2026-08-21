# Step 8 — Candidate Composite Reliability Formulations

We define 3 candidate mathematical formulations for a composite reliability index ($R \in [0, 1]$):

---

## 1. Formulation A: Equal-Weight Metric Composite
Average of 6 normalized metric components:
$$R_A = \frac{1}{6} \left( \text{MacroF1} + \text{LGGRec} + \text{AUC} + (1 - \text{ECE}) + (1 - \text{Brier}) + (1 - \text{HighConfErrRate}) \right)$$

- **Strengths**: Unbiased, simple, easy to reproduce.
- **Weaknesses**: Gives equal weight to metrics with high conceptual overlap (Macro-F1 & LGG Recall).

---

## 2. Formulation B: Dimension-Balanced Composite
Equal weight across 3 core functional dimensions (Discrimination, Calibration, Safety):
$$\text{Discrimination} = \frac{1}{3}(\text{MacroF1} + \text{LGGRec} + \text{AUC})$$
$$\text{Calibration} = \frac{1}{2}((1 - \text{ECE}) + (1 - \text{Brier}))$$
$$\text{Safety} = (1 - \text{HighConfErrRate})$$
$$R_B = \frac{1}{3} \text{Discrimination} + \frac{1}{3} \text{Calibration} + \frac{1}{3} \text{Safety}$$

- **Strengths**: Structurally prevents any single dimension from dominating the composite score.
- **Weaknesses**: Does not prioritize minority clinical risk over general calibration.

---

## 3. Formulation C: Clinically-Weighted Composite
Explicitly prioritizes minority LGG sensitivity and overconfidence safety:
$$R_C = 0.20 \cdot \text{MacroF1} + 0.25 \cdot \text{LGGRec} + 0.15 \cdot \text{AUC} + 0.15 \cdot (1 - \text{ECE}) + 0.10 \cdot (1 - \text{Brier}) + 0.15 \cdot (1 - \text{HighConfErrRate})$$

- **Strengths**: Reflects clinical priorities in neuro-oncology (protecting LGG patients from misclassification toxicity).
- **Weaknesses**: Weights introduce subjective choice, requiring explicit justification.
