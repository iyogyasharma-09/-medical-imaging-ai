# Manuscript Draft: Section 4 — Discussion Outline

### A. Insufficiency of Single Aggregate Metrics in Medical Image Classification
- Aggregate accuracy and ROC-AUC are easily dominated by majority-class volume under severe clinical imbalance (e.g. 93.6% HGG).
- High AUC guarantees good case ranking, but provides zero guarantee that standard decision rules will diagnose rare pathologies.

### B. Decoupling of the Three Core Dimensions
- **Discrimination (Ranking):** Governed by learned feature representations; invariant to monotonic transformations and threshold cutoffs.
- **Calibration (Statistical Reliability):** Governed by logit scaling and loss formulation; repairable post-hoc via validation temperature scaling without retraining.
- **Decision Boundary (Operating Point):** Governed by clinical utility, risk tolerance, and threshold placement; adjustable post-hoc without altering model weights.

### C. The Focal Loss Paradox & Post-Hoc Resolution
- Loss functions designed to counteract class imbalance can introduce unexpected posterior distortions (probability compression).
- Rather than declaring Focal Loss a 'failure', recognizing this as a calibration artifact allows post-hoc temperature scaling to restore statistical reliability while leveraging the learned ranking manifold.

### D. Architectural Sensitivity to Loss Interventions
- 3D CNNs (ResNet, DenseNet) and Vision Transformers (SwinUNETR) responded differently to identical class weighting protocols.
- Highlights that loss re-weighting is not a plug-and-play universal remedy; architectural inductive biases influence loss landscape navigation under limited sample regimes.

### E. Clinical Deployment Recommendations
1. Report class-stratified sensitivity and specificity alongside aggregate metrics.
2. Verify probability calibration (ECE/Brier) and apply validation-only temperature scaling when using margin or focal losses.
3. Pre-specify operational decision thresholds on validation cohorts using task-specific utility functions rather than defaulting to 0.50.

### F. Preserved Scientific Limitations
- Small sample size of minority pathology (12 locked-test LGG subjects).
- Single-seed canonical training runs.
- Descriptive nature of locked-test operating curve sweeps.
- Internal mechanisms of architectural inductive bias were not directly investigated.

### G. Future Directions
- Multi-seed stochastic variance isolation across diverse medical domains.
- Patient-specific morphometric uncertainty modeling.
- Prospective clinical utility optimization in multi-center deployment trials.
