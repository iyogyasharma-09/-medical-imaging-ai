# Decision-Threshold Sensitivity: Methodology

**Project**: BraTS 2023 Adult Glioma (GLI) 3D Grading Pipeline  
**Experiment**: Post-Hoc Operating-Point & Threshold Sensitivity Analysis  
**Cohort Evaluated**:
- **Development Selection Set**: 187 Validation Subjects (10 LGG, 177 HGG) from `BRATS2023_VAL_MANIFEST`
- **Locked Test Evaluation Set**: 188 Test Subjects (12 LGG, 176 HGG) from `BRATS2023_TEST_MANIFEST`
**Audit Date**: 2026-08-17  
**Audit Mode**: STRICT READ-ONLY ANALYSIS  

---

## 1. Decision Threshold Mathematical Formulation

In binary grading between minority class 0 (LGG) and majority class 1 (HGG), neural networks output continuous class probabilities via Softmax:

$$P(\text{LGG}) = p_0, \quad P(\text{HGG}) = p_1, \quad p_0 + p_1 = 1.0$$

The standard argmax classification rule uses a fixed decision threshold:

$$\hat{y}_{\text{default}} = \begin{cases} 0 \ (\text{LGG}) & \text{if } P(\text{LGG}) \ge 0.50 \\ 1 \ (\text{HGG}) & \text{if } P(\text{LGG}) < 0.50 \end{cases}$$

Under severe class imbalance (95% HGG : 5% LGG), standard training objectives frequently produce uncalibrated probabilities skewed toward the majority class. While the network may correctly rank LGG cases above HGG cases (high ROC-AUC), predicted probabilities for true LGG cases may fall into the range $[0.10, 0.45]$, causing argmax at $0.50$ to classify them as HGG (producing low or zero LGG recall).

### Generalized Operating-Point Parameter:
Let $\tau_{\text{LGG}} \in [0.01, 0.99]$ be the decision threshold on the minority class probability:

$$\hat{y}(\tau_{\text{LGG}}) = \begin{cases} 0 \ (\text{LGG}) & \text{if } P(\text{LGG}) \ge \tau_{\text{LGG}} \\ 1 \ (\text{HGG}) & \text{if } P(\text{LGG}) < \tau_{\text{LGG}} \end{cases}$$

Lowering $\tau_{\text{LGG}}$ increases sensitivity to the minority class (LGG Recall) at the cost of decreasing specificity (HGG Recall).

---

## 2. Threshold Selection Rules on Development Data

To prevent test-set leakage, all optimal thresholds $\tau^*$ were selected **strictly on development validation data** ($N=187$) and then applied **once out-of-sample** to the locked test set ($N=188$).

### Evaluated Selection Rules:

1. **Rule D (Default Operating Point)**:
   $$\tau = 0.50$$
   The standard baseline used across Day 1–5 evaluations.

2. **Rule A (Maximum Validation Macro-F1)**:
   $$\tau^*_A = \arg\max_{\tau \in [0.01, 0.99]} \text{Macro-F1}_{\text{val}}(\tau)$$
   Optimizes the unweighted arithmetic mean of class-0 F1 and class-1 F1.

3. **Rule B (Maximum LGG Recall with Clinical Specificity Constraint $\ge 90\%$)**:
   $$\tau^*_B = \arg\max_{\tau} \{\text{LGG\_Recall}_{\text{val}}(\tau) \mid \text{HGG\_Recall}_{\text{val}}(\tau) \ge 0.90\}$$
   Maximizes detection of the minority grade while enforcing that false-positive LGG alarms do not exceed 10% on the majority class.

4. **Rule C (Maximum Balanced Accuracy / Youden's J Index)**:
   $$\tau^*_C = \arg\max_{\tau \in [0.01, 0.99]} \frac{\text{LGG\_Recall}_{\text{val}}(\tau) + \text{HGG\_Recall}_{\text{val}}(\tau)}{2}$$
   Finds the optimal point on the ROC curve maximizing the sum of sensitivity and specificity.

---

## 3. Methodological Audit of Earlier 0.10 / 0.17 Thresholds

An audit of `results/advanced_analysis.json` (lines 32–60) investigated previously reported thresholds (ResNet18: 0.10, DenseNet121: 0.17, SwinUNETR: 0.10):

1. **Checkpoint Identity**: Those thresholds were generated from an early exploratory script operating on transient 7-epoch and 11-epoch checkpoints that were subsequently overwritten during protocol development.
2. **Objective Flaw**: The script optimized `val_best_f1` using majority-weighted / binary HGG F1 ($\approx 0.92 - 0.95$), rather than Macro-F1. For ResNet18 and SwinUNETR, setting $\theta = 0.10$ on $P(\text{HGG})$ forced near-unconditional HGG predictions, producing `test_specificity = 0.0` (0% LGG recall).
3. **Verdict**: The earlier 0.10/0.17 thresholds are **methodologically obsolete**. All subsequent analysis relies strictly on the rigorous development rules A, B, and C defined above.
