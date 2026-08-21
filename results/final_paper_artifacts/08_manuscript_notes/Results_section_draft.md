# 3. Experimental Results

## 3.1 Dataset and Experimental Evaluation Design
The empirical evaluation was conducted on the multi-parametric 3D magnetic resonance imaging (mpMRI) cohort from the Brain Tumor Segmentation (BraTS) 2023 Adult Glioma (GLI) benchmark. The study evaluated the binary diagnostic task of classifying low-grade gliomas (LGG, grade 2/3, label 0) from high-grade gliomas (HGG, grade 4 glioblastoma, label 1) using standardized four-channel 3D volumes (T1, T1-contrast, T2, and FLAIR) normalized and resampled to $4 \times 96 \times 96 \times 96$ voxels. 

The cohort was partitioned into fixed, deterministic splits: a training split of $n = 876$ subjects, an independent validation split of $n = 187$ subjects ($10$ LGG, $177$ HGG; $5.3\%$ minority prevalence), and a locked, read-only test split of $n = 188$ subjects ($12$ LGG, $176$ HGG; $6.4\%$ minority prevalence). Preprocessing transformations were held constant across all experiments using deterministic foreground cropping, axial re-orientation, and channel-wise non-zero z-score intensity normalization. All post-hoc calibration parameters were optimized strictly on the validation split and subsequently evaluated on the locked test cohort without test-set label feedback.

## 3.2 Baseline Discrimination and Class-Stratified Performance
We first evaluated three canonical 3D deep learning architectures under standard, unweighted Cross-Entropy loss (Protocol A): a 3D Residual Network initialized with MedicalNet pre-trained transfer weights (ResNet18 MedicalNet), a 3D Densely Connected Network (DenseNet121 3D), and a Hierarchical 3D Vision Transformer (SwinUNETR 3D) (Table 1).

On aggregate metrics, all three architectures demonstrated high classification accuracy and area under the receiver operating characteristic curve (ROC-AUC) on the locked test cohort: ResNet18 achieved an accuracy of $97.34\%$ with $\text{ROC-AUC} = 0.9830$; DenseNet121 achieved $94.68\%$ accuracy with $\text{ROC-AUC} = 0.9422$; and SwinUNETR achieved $93.62\%$ accuracy with $\text{ROC-AUC} = 0.9330$. 

However, class-stratified evaluation revealed marked disparities in minority-class diagnostic sensitivity at the standard decision threshold ($\tau = 0.50$). While ResNet18 and DenseNet121 achieved LGG recalls of $75.00\%$ ($9/12$) and $66.67\%$ ($8/12$) alongside HGG recalls of $98.86\%$ ($174/176$) and $96.59\%$ ($170/176$) respectively (yielding Macro-F1 scores of $0.8842$ and $0.7934$), SwinUNETR exhibited complete sensitivity suppression on the minority class ($0.00\%$ LGG recall, $0/12$; Macro-F1 = $0.4835$). Despite ranking test subjects with high discriminatory capacity ($\text{ROC-AUC} = 0.9330$), SwinUNETR predicted the majority class for all 188 test subjects at the default $\tau = 0.50$ cutoff. These baseline results demonstrate that aggregate accuracy and ROC-AUC can coexist with severe minority-class sensitivity limitations in clinical grading.

## 3.3 Effect of Class-Imbalance Handling Strategies
To evaluate whether loss-level re-weighting resolves minority-class sensitivity limitations during training, we evaluated two loss formulations across all three architectures: inverse-frequency class weighting (Protocol B) and adjusted minority weighting (Protocol C) (Table 1).

The empirical response to class weighting was divergent and architecture-dependent:
1. For SwinUNETR, loss re-weighting partially mitigated minority suppression. Under Protocol C, SwinUNETR improved from $0.00\%$ to $25.00\%$ LGG recall ($3/12$) while maintaining $99.43\%$ HGG recall ($175/176$), increasing Macro-F1 from $0.4835$ to $0.6736$ and achieving $\text{ROC-AUC} = 0.9735$.
2. Conversely, for ResNet18 and DenseNet121, class weighting reduced minority sensitivity. ResNet18 LGG recall decreased from $75.00\%$ (Protocol A) to $66.67\%$ (Protocol B) and $41.67\%$ (Protocol C), while Macro-F1 decreased from $0.8842$ to $0.7267$ and $0.6884$. Similarly, DenseNet121 LGG recall decreased from $66.67\%$ (Protocol A) to $58.33\%$ (Protocol B) and $33.33\%$ (Protocol C).

These findings demonstrate that static loss re-weighting does not produce consistent sensitivity improvements across distinct deep learning architectures under extreme class imbalance. The underlying mechanisms governing these architectural interactions were not directly investigated and represent an important direction for future research.

## 3.4 Threshold-Dependent Operating Behaviour
To determine whether low minority-class recall at $\tau = 0.50$ reflected a deficit in learned feature representations or a misalignment of the default decision cutoff, we conducted decision threshold sensitivity analyses across calibrated output probabilities (Table 4, Figure 3).

Across all evaluated models, decision threshold sweeps demonstrated continuous sensitivity/specificity trade-off manifolds. For the ResNet18 Focal Loss model ($\text{ROC-AUC} = 0.9541$), adjusting the decision threshold on calibrated LGG probability ($P_{\text{cal}}(\text{LGG}) \geq \tau_{\text{LGG}}$) yielded the following observed operating points:
- **Default Operating Point ($\tau = 0.50$):** LGG Recall = $41.67\%$ ($5/12$), HGG Recall = $97.16\%$ ($171/176$), Macro-F1 = $0.7103$, Balanced Accuracy = $0.6941$.
- **Maximum Macro-F1 Operating Point ($\tau = 0.35$):** LGG Recall = $66.67\%$ ($8/12$), HGG Recall = $95.45\%$ ($168/176$), Macro-F1 = $0.7685$, Balanced Accuracy = $0.8106$.
- **High-Sensitivity Operating Point ($\tau = 0.20$):** LGG Recall = $83.33\%$ ($10/12$), HGG Recall = $88.64\%$ ($156/176$), Macro-F1 = $0.7052$, Balanced Accuracy = $0.8598$.
- **Maximum Minority-Recall Operating Point ($\tau = 0.10$):** LGG Recall = $100.00\%$ ($12/12$), HGG Recall = $83.52\%$ ($147/176$), Macro-F1 = $0.6815$, Balanced Accuracy = $0.9176$.

Critically, shifting the decision threshold across the probability domain produced zero change in ROC-AUC ($0.9541$), which remains mathematically invariant to threshold selection. These results demonstrate that ranking discrimination and operational decision boundaries represent distinct evaluative dimensions; models that exhibit modest sensitivity at $\tau = 0.50$ may possess viable high-sensitivity operating regimes on their calibrated probability scale.

## 3.5 Focal Loss Intervention
We evaluated Focal Loss ($\alpha = 1.0, \gamma = 2.0$) as a dynamic hardness-modulating intervention on ResNet18 MedicalNet (Table 2). Training converged with early stopping at epoch 14, attaining a validation Macro-F1 of $0.7225$ (validation LGG recall = $50.00\%$).

On the locked test cohort, Focal Loss retained high discriminatory ranking capacity with $\text{ROC-AUC} = 0.9541$, achieving an overall classification accuracy of $93.62\%$ and Macro-F1 of $0.7103$. At the default decision boundary ($\tau = 0.50$), the model achieved $41.67\%$ LGG recall ($5/12$) and $97.16\%$ HGG recall ($171/176$). Inspection of probability calibration revealed that Focal Loss produced an uncalibrated Expected Calibration Error (ECE) of $9.75\%$ ($0.0975$) and a Brier score of $0.0497$. Modulating easy majority samples during optimization attenuated logit magnitudes, compressing predicted LGG probabilities into a sub-unity range ($[0.00, 0.73]$). Consequently, Focal Loss retained strong discriminative capacity, but its default-threshold operational performance was constrained by probability scale compression.

## 3.6 Post-Hoc Probability Calibration
To decompress the probability distribution without altering learned feature representations, we applied post-hoc temperature scaling. A single scalar parameter ($T^*$) was optimized strictly by minimizing Negative Log-Likelihood (NLL) on the 187-subject validation cohort, yielding $T^* = 0.4923$ (reducing validation NLL from $0.2099$ to $0.1674$ and validation ECE from $8.29\%$ to $1.75\%$).

When applied to the locked test cohort with frozen $T^*$, temperature scaling reduced test ECE by $65.0\%$, from $9.75\%$ to $3.41\%$ ($0.0341$), and reduced the test Brier score from $0.0497$ to $0.0394$ (Table 2, Table 3). Because scalar temperature scaling is a strictly monotonic logit transformation, ROC-AUC remained exactly $0.9541$ ($\Delta \text{AUC} = 0.0000000000$), and binary classifications at the default cutoff ($\tau = 0.50$) were $100\%$ identical (LGG recall = $41.67\%$, HGG recall = $97.16\%$). Post-hoc temperature scaling substantially improved posterior probability reliability, while preserving the underlying case rankings and default-threshold classifications.

## 3.7 Cross-Architecture Calibration Comparison
To evaluate whether temperature scaling provides consistent calibration behavior across model families, we benchmarked post-hoc calibration across all nine baseline checkpoints and the Focal Loss intervention (Table 3, Figure 2).

The empirical benchmark revealed distinct baseline calibration dynamics:
1. Models trained with standard Cross-Entropy (Protocol A) exhibited baseline temperature parameters close to unity ($T^* \in [1.023, 1.210]$) and low baseline test ECEs ($2.76\% - 3.45\%$), reflecting the log-likelihood optimization objective of Cross-Entropy. Post-hoc scaling provided modest refinements (e.g., SwinUNETR Protocol A test ECE decreased from $3.45\%$ to $1.12\%$, $T^* = 1.2101$).
2. In contrast, models subjected to loss hardness modifications (Focal Loss) exhibited substantial logit compression ($T^* = 0.4922$) and derived large calibration benefits ($65.0\%$ ECE reduction, $9.75\% \to 3.41\%$).
3. Importantly, high calibration quality did not guarantee minority sensitivity. For example, SwinUNETR under Protocol A achieved a calibrated ECE of $1.12\%$, yet diagnosed $0.00\%$ of minority cases at default $\tau = 0.50$. Similarly, ResNet18 under Protocol C achieved a calibrated ECE of $0.99\%$, yet achieved only $41.67\%$ LGG recall at default cutoff.

These comparisons demonstrate that probability calibration evaluates the statistical reliability of confidence scores, but does not dictate whether the chosen decision threshold satisfies clinical sensitivity requirements.

## 3.8 Integrated Trustworthy-Performance Assessment
Synthesizing results across all ten experimental configurations demonstrates that clinical model reliability cannot be characterized by a single aggregate metric. Rather, our experiments demonstrate that four critical evaluative dimensions frequently decouple under severe class imbalance:
1. **Rank Discrimination (ROC-AUC):** Reflects feature separability across the cohort ($0.8667 - 0.9830$) and remains invariant to monotonic calibration and threshold adjustments.
2. **Class-Stratified Sensitivity:** Measures clinical safety for rare disease subtypes; unmasked severe sensitivity disparities ($0.00\% - 75.00\%$) that were concealed by aggregate accuracy ($>91.49\%$).
3. **Probability Calibration (ECE / Brier):** Quantifies whether predicted probabilities reflect empirical event rates ($0.99\% - 5.23\%$); directly repairable post-hoc via validation temperature scaling without retraining.
4. **Operational Decision Boundaries:** Dictates the clinical sensitivity/specificity operating point along the calibrated probability axis (yielding up to $100\%$ sensitivity on descriptive sweeps).
5. **Predictive Uncertainty:** Subject-level predictive entropy and calibrated confidence identify borderline and high-risk cases near decision boundaries.

Together, these findings provide empirical support for an integrated evaluation framework that measures discrimination, calibration, class-stratified sensitivity, and decision-threshold behavior as complementary pillars of trustworthy clinical AI.
