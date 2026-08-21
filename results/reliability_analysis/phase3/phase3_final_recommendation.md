# Step 12 — Final Methodological Recommendations

## 1. Project-Specific Definition of Reliability

In the BraTS 2023 3D GLI Grading Pipeline, a model is defined as **Reliable** if and only if it satisfies four simultaneous criteria:
1. **High Discrimination**: Achieves strong balanced grading performance (Macro-F1 $\ge 0.75$).
2. **Minority Protection**: Maintains high sensitivity to minority LGG cases (LGG Recall $\ge 0.65$).
3. **Calibrated Probabilities**: Produces probability estimates that accurately reflect true accuracy ($ECE \le 0.05$).
4. **Overconfidence Safety**: Avoids severe high-confidence false predictions ($	ext{Conf} \ge 0.80$ error rate $\le 0.02$).

---

## 2. Final Paper Reporting Recommendation

We recommend **Option A + Option B Dual Reporting Structure**:
1. **Primary Ground Truth**: Report the full **Multi-Dimensional Reliability Profile** (Macro-F1, LGG Recall, AUC, ECE, Brier Score, High-Confidence Error Count) in a comprehensive dashboard table.
2. **Secondary Ranking Metric**: Report **Formulation B (Dimension-Balanced Composite Index)** as a single scalar summary, gated by a mandatory clinical safety check (models with LGG Recall $< 0.50$ are flagged as clinically non-viable).

---

## 3. Canonical Winner Confirmation
Under all formulations, profiles, and individual metrics:
**ResNet18 MedicalNet (Protocol A)** is confirmed as the **canonical winning model** of the project.
