# 4. Results

## 4.1 Locked Test Performance
We evaluated nine canonical model/protocol configurations on the frozen BraTS 2023 test set (188 GLI subjects: 12 LGG / 176 HGG). ResNet18 MedicalNet trained under unweighted CrossEntropyLoss (Protocol A) achieved the top overall performance, reaching a Test Macro-F1 of 0.8842, Test Accuracy of 0.9734, Test LGG Recall of 0.7500 (9/12 LGG detected), and Test AUC of 0.9830. DenseNet121 3D (Protocol A) followed with a Macro-F1 of 0.7934 and LGG Recall of 0.6667. SwinUNETR 3D under Protocol A collapsed onto the majority class (0.0% LGG Recall, Macro-F1 0.4835).

## 4.2 Calibration and Reliability Analysis
Expected Calibration Error (ECE) and Brier Score were calculated across all configurations. ResNet18 MedicalNet (Protocol A) demonstrated superior probability calibration, obtaining an ECE of 0.0276 and Brier Score of 0.0496, committing only a single high-confidence error ($\ge 0.80$). In contrast, applying aggressive class weighting (Protocol C) degraded ResNet18 calibration (ECE 0.0142, Brier 0.1019, 6 high-confidence errors).

## 4.3 Reliability Score Analysis
Using the proposed Dimension-Balanced Reliability Index ($R_B$), ResNet18 MedicalNet (Protocol A) achieved the highest overall score ($R_B = 0.9298$), followed by DenseNet121 3D Protocol A ($R_B = 0.8938$). Clinical safety gating ($	ext{LGG Recall} \ge 0.50$) disqualified 6 out of 9 configurations due to insufficient minority class sensitivity.

## 4.4 Class-Weighting Ablation
Ablation Study #1 evaluated loss weighting transitions ($A 	o B$, $B 	o C$, $A 	o C$). For ResNet18 MedicalNet, transitioning from unweighted Protocol A to strong minority weighting Protocol C reduced Macro-F1 by 0.1958 and LGG Recall by 0.3333. A similar deterioration occurred for DenseNet121 3D ($\Delta 	ext{F1} = -0.1407$, $\Delta 	ext{LGGRec} = -0.3334$). Conversely, SwinUNETR 3D required Protocol C weighting to achieve non-zero minority detection ($\Delta 	ext{F1} = +0.1901$, $\Delta 	ext{LGGRec} = +0.2500$).

## 4.5 Architecture-Dependent Effects
The observed results indicate that the benefit of class weighting is strongly architecture-dependent. Pretrained 3D CNNs with transfer learning (MedicalNet) form robust decision boundaries without loss weighting, whereas loss weighting distorts their probability calibration. Un-pretrained vision transformers (SwinUNETR 3D) lack spatial priors and require class weighting to prevent majority-class collapse.

## 4.6 Main Findings
1. ResNet18 MedicalNet (Protocol A) is the top-performing and most reliable model configuration.
2. Class weighting does not universally improve minority classification; its effect is architecture-dependent.
3. High ROC-AUC does not guarantee calibration or clinical safety.
