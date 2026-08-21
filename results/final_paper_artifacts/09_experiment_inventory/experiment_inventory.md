# Definitive Experiment Inventory

This document catalogues all 10 completed experimental stages in the BraTS 2023 GLI trustworthy evaluation research project.

### EXP-01: Standard Cross-Entropy Baseline (Protocol A)
- **Research Question:** How do standard 3D CNNs and Vision Transformers perform under default unweighted Cross-Entropy loss?
- **Models & Architectures:** ResNet18 MedicalNet, DenseNet121 3D, SwinUNETR 3D
- **Protocol:** Protocol A (Unweighted CE, lr=1e-4, batch=2, seed=42)
- **Cohorts:** Val: 187 (10 LGG, 177 HGG); Test: 188 (12 LGG, 176 HGG)
- **Training Occurred:** Yes
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Accuracy, Macro-F1, LGG/HGG Recall, AUC, ECE, Brier
- **Output Directory:** [`results/protocol_archives/protocol_A/`](file:///C:\mri\mri-\results\protocol_archives\protocol_A)
- **Main Finding:** High aggregate accuracy (>93.6%) and AUC (>0.93) masked extreme minority-class recall disparity (Swin: 0.0% LGG recall; ResNet: 75.0%).
- **Preserved Limitation:** Single-seed training; severe class imbalance (93.6% majority).

### EXP-02: Class-Weighting Interventions (Protocols B & C)
- **Research Question:** Does loss re-weighting (inverse-frequency Protocol B and adjusted minority Protocol C) reliably resolve minority-class sensitivity?
- **Models & Architectures:** ResNet18 MedicalNet, DenseNet121 3D, SwinUNETR 3D
- **Protocol:** Protocol B (Inv-Freq), Protocol C (Adjusted Minority)
- **Cohorts:** Val: 187 (10 LGG, 177 HGG); Test: 188 (12 LGG, 176 HGG)
- **Training Occurred:** Yes
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Accuracy, Macro-F1, LGG/HGG Recall, AUC, ECE, Brier
- **Output Directory:** [`results/protocol_archives/protocol_B/, protocol_C/`](file:///C:\mri\mri-\results\protocol_archives\protocol_B\, protocol_C)
- **Main Finding:** Produced divergent, architecture-dependent responses. Swin improved (0% -> 25%), while ResNet degraded (75% -> 41.7%).
- **Preserved Limitation:** Mechanism of architecture-specific weighting response was not causally isolated.

### EXP-03: Uncalibrated Baseline Threshold Sensitivity Analysis
- **Research Question:** Does the fixed tau=0.50 decision boundary mischaracterize model discriminative capabilities across architectures?
- **Models & Architectures:** ResNet18, DenseNet121, SwinUNETR (Protocols A, B, C)
- **Protocol:** Post-hoc sweep tau in [0.05, 0.95]
- **Cohorts:** Val: 187; Test: 188
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** LGG/HGG Recall, Balanced Acc, Macro-F1 vs tau
- **Output Directory:** [`results/reliability_analysis/threshold_analysis/`](file:///C:\mri\mri-\results\reliability_analysis\threshold_analysis)
- **Main Finding:** Altering decision threshold exposed continuous sensitivity trade-offs while ROC-AUC remained completely invariant.
- **Preserved Limitation:** Threshold sweeps on test cohort are descriptive operating curves, not prospective selection.

### EXP-04: Focal Loss Intervention on ResNet18 MedicalNet
- **Research Question:** Does dynamic sample-hardness weighting (Focal Loss alpha=1.0, gamma=2.0) resolve minority sensitivity without heuristic re-weighting?
- **Models & Architectures:** ResNet18 MedicalNet
- **Protocol:** Focal Loss (alpha=1.0, gamma=2.0, lr=1e-4, batch=2, seed=42)
- **Cohorts:** Val: 187; Test: 188
- **Training Occurred:** Yes
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Val Macro-F1, LGG Recall, Checkpoint State
- **Output Directory:** [`models/resnet18_medicalnet_focal_lgg_hgg_3d.pt`](file:///C:\mri\mri-\models\resnet18_medicalnet_focal_lgg_hgg_3d.pt)
- **Main Finding:** Model trained stably (early stopping epoch 14), achieving strong validation F1 (0.7225) and LGG recall (50%).
- **Preserved Limitation:** Single canonical seed (seed 42); evaluated on ResNet18 backbone.

### EXP-05: Locked-Test Evaluation of Focal Loss Checkpoint
- **Research Question:** How does Focal Loss perform on the identical locked test cohort compared to the CE baseline?
- **Models & Architectures:** ResNet18 MedicalNet (Focal)
- **Protocol:** Deterministic locked-test evaluation (eval mode)
- **Cohorts:** Test: 188 (12 LGG, 176 HGG)
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Acc=0.9362, F1=0.7103, AUC=0.9541, LGG Rec=0.4167, HGG Rec=0.9716, ECE=0.0975, Brier=0.0497
- **Output Directory:** [`results/focal_locked_test/`](file:///C:\mri\mri-\results\focal_locked_test)
- **Main Finding:** Retained high rank discrimination (AUC=0.9541), but default tau=0.50 recall remained limited (41.67%) with noticeable probability compression (ECE=9.75%).
- **Preserved Limitation:** Revealed probability compression under default thresholding.

### EXP-06: Post-Hoc Validation-Only Temperature Scaling of Focal Loss
- **Research Question:** Can validation-fitted temperature scaling correct Focal Loss probability compression without altering discrimination?
- **Models & Architectures:** ResNet18 MedicalNet (Focal)
- **Protocol:** Scalar temperature T* optimized strictly on val NLL; frozen before test eval
- **Cohorts:** Val: 187 (for T* fit); Test: 188 (for eval)
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** T*=0.4923, Test ECE: 9.75% -> 3.41%, Brier: 0.0497 -> 0.0394, AUC=0.9541 (Invariant)
- **Output Directory:** [`results/focal_locked_test/temperature_scaling/`](file:///C:\mri\mri-\results\focal_locked_test\temperature_scaling)
- **Main Finding:** Test ECE was reduced by 65% (matching CE baseline of 3.69%) while ROC-AUC and tau=0.50 classifications remained strictly invariant.
- **Preserved Limitation:** Temperature scaling corrects probability scale, but does not alter default binary classifications.

### EXP-07: Calibrated Focal Operating-Point Analysis
- **Research Question:** How do calibrated probabilities translate into clinical operating points across decision thresholds?
- **Models & Architectures:** ResNet18 MedicalNet (Focal + T*=0.4923)
- **Protocol:** Threshold sweep tau in [0.10, 0.90] on calibrated probabilities
- **Cohorts:** Test: 188
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** tau=0.35 -> F1=0.7685 (LGG Rec=66.7%, HGG Rec=95.5%); tau=0.10 -> LGG Rec=100%, HGG Rec=83.5%
- **Output Directory:** [`results/focal_locked_test/temperature_scaling/threshold_analysis/`](file:///C:\mri\mri-\results\focal_locked_test\temperature_scaling\threshold_analysis)
- **Main Finding:** Operating on calibrated probabilities provides principled sensitivity trade-offs (e.g. 66.7% LGG recall with only 4 false alarms).
- **Preserved Limitation:** Descriptive analysis; operational cutoff selection requires prospective validation.

### EXP-08: Multi-Model Post-Hoc Calibration Benchmark
- **Research Question:** Does post-hoc temperature scaling generalize across CNNs and Vision Transformers under Protocols A, B, and C?
- **Models & Architectures:** 9 Baseline models (Protocols A, B, C x 3 archs) + Focal ResNet18
- **Protocol:** Validation NLL minimization for T*, evaluated on locked test
- **Cohorts:** Val: 187; Test: 188
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Multi-model NLL, Brier, Confidence ECE, Prob ECE, AUC, Recalls
- **Output Directory:** [`results/posthoc_calibration_comparison/`](file:///C:\mri\mri-\results\posthoc_calibration_comparison)
- **Main Finding:** CE models are already well-calibrated (T* ~ 1.0, ECE 1.4-3.5%); Focal model derived large benefit (T*=0.49, ECE 9.8% -> 3.4%).
- **Preserved Limitation:** Evaluated on single-temperature parameter scaling.

### EXP-09: Multi-Model Calibrated Threshold Sensitivity Benchmark
- **Research Question:** How do decision boundaries interact with calibrated probabilities across diverse architectures?
- **Models & Architectures:** All 10 models (9 baselines + Focal)
- **Protocol:** Calibrated probability threshold sweeps tau in [0.10, 0.90]
- **Cohorts:** Test: 188
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Operating points at tau in [0.10, 0.90], Max F1, Max BalAcc
- **Output Directory:** [`results/posthoc_calibration_comparison/multi_model_threshold_comparison.csv`](file:///C:\mri\mri-\results\posthoc_calibration_comparison\multi_model_threshold_comparison.csv)
- **Main Finding:** Demonstrated that discrimination, calibration, and operating threshold choices frequently decouple across models.
- **Preserved Limitation:** Threshold sweeps reflect locked-test empirical distributions.

### EXP-10: Predictive Uncertainty & Reliability Analysis
- **Research Question:** Can predictive entropy and confidence identify borderline, high-risk cases near decision boundaries?
- **Models & Architectures:** All 10 models
- **Protocol:** Entropy and confidence computed per subject from calibrated probabilities
- **Cohorts:** Test: 188
- **Training Occurred:** No (Post-Hoc Analysis)
- **Locked-Test Isolation:** Maintained (Zero test fitting)
- **Key Metrics:** Predictive Entropy, Calibrated Confidence per subject
- **Output Directory:** [`results/posthoc_calibration_comparison/calibrated_predictions/`](file:///C:\mri\mri-\results\posthoc_calibration_comparison\calibrated_predictions)
- **Main Finding:** High predictive entropy correlates strongly with misclassified and borderline minority cases.
- **Preserved Limitation:** Epistemic vs aleatoric uncertainty separation is approximated.

