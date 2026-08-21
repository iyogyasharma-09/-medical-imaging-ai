# 2. Materials and Methods

## 2.1 Study Design and Experimental Overview
This study establishes a controlled evaluation framework designed to characterize the multidimensional reliability of 3D deep learning models for medical image classification under severe class imbalance. Rather than evaluating algorithmic performance through a single summary statistic, our experimental design systematically disaggregates five complementary evaluative dimensions:
1. **Ranking Discrimination:** Quantifying continuous feature separability across disease classes via rank-based metrics ($\text{ROC-AUC}$).
2. **Class-Stratified Diagnostic Sensitivity:** Quantifying clinical safety for rare pathological subtypes through separate low-grade glioma (LGG) and high-grade glioma (HGG) recalls.
3. **Probability Calibration:** Quantifying the statistical fidelity of predicted confidence scores relative to observed empirical frequencies ($\text{ECE}$ and Brier score).
4. **Operational Decision Boundaries:** Characterizing sensitivity/specificity trade-off manifolds along the predicted probability scale via decision threshold sweeps ($\tau$).
5. **Predictive Uncertainty:** Quantifying sample-level diagnostic ambiguity via information-theoretic predictive entropy.

The overall experimental workflow comprises: (i) cohort curation from the BraTS 2023 benchmark; (ii) standardized 3D multi-parametric MRI preprocessing; (iii) deterministic partitioning into training ($n=876$), validation ($n=187$), and locked test ($n=188$) splits; (iv) controlled training of three distinct 3D deep learning architectures under standard, inverse-frequency weighted, and minority-emphasized Cross-Entropy protocols; (v) a dedicated Focal Loss intervention on 3D ResNet18; (vi) read-only locked-test evaluation; (vii) validation-only optimization of post-hoc temperature scaling parameters ($T^*$); (viii) locked-test evaluation of calibrated probabilities; and (ix) predictive uncertainty profiling.

## 2.2 Dataset and Cohort Construction
All experiments were conducted on multi-parametric 3D MRI volumes from the Brain Tumor Segmentation (BraTS) 2023 Adult Glioma (GLI) Challenge dataset. The clinical objective was binary histopathological grade classification: distinguishing low-grade gliomas (LGG, WHO grade 2/3, encoded as $y=0$) from high-grade glioblastoma (HGG, WHO grade 4, encoded as $y=1$), defined by the primary `label_grade` target column.

The complete cohort was partitioned into fixed, deterministic splits with zero subject overlap:
- **Training Cohort ($n = 876$):** Utilized exclusively for gradient-based model parameter optimization.
- **Validation Cohort ($n = 187$):** Comprising $10$ LGG ($5.3\%$) and $177$ HGG ($94.7\%$) cases; utilized exclusively for intra-training checkpoint selection via early stopping and post-hoc temperature scaling optimization ($T^*$).
- **Locked Test Cohort ($n = 188$):** Comprising $12$ LGG ($6.4\%$) and $176$ HGG ($93.6\%$) cases; held strictly read-only and evaluated only in forward evaluation mode with frozen model and calibration weights.

Cohort split manifests were generated using fixed random seed initialization (`RANDOM_SEED = 42`) and persisted as deterministic JSON manifests (`train_manifest.json`, `val_manifest.json`, `test_manifest.json`).

## 2.3 MRI Modalities and Input Representation
Each subject volume comprised four co-registered 3D multi-parametric MRI sequences:
1. Native T1-weighted ($\text{T1}$)
2. Post-contrast T1-weighted ($\text{T1ce}$)
3. T2-weighted ($\text{T2}$)
4. T2 Fluid-Attenuated Inversion Recovery ($\text{FLAIR}$)

Input volumes were formatted as 4-channel 3D tensors with channel order $[\text{T1}, \text{T1ce}, \text{T2}, \text{FLAIR}]$ along the leading dimension (channel index $C=0$). The standardized spatial dimension for all inputs was $4 \times 96 \times 96 \times 96$ voxels ($C \times D \times H \times W$), selected to ensure deterministic batch execution within dedicated GPU memory constraints ($8\text{ GB}$ VRAM).

## 2.4 Preprocessing and Augmentation Pipelines
Preprocessing transformations were implemented using the MONAI framework to enforce strict geometric and radiometric consistency.

### 2.4.1 Deterministic Validation and Test Pipeline
Validation and locked-test volumes were processed through an identical, deterministic evaluation pipeline:
1. **Channel Alignment (`EnsureChannelFirstd`):** Enforces leading channel dimension without inserting singleton axes ($4, D, H, W$).
2. **Anatomical Reorientation (`Orientationd`):** Reorients all volumes to canonical Right-Anterior-Superior ($\text{RAS}$) coordinate space.
3. **Isotropic Resampling (`Spacingd`):** Resamples spatial voxels to $1.0\text{ mm} \times 1.0\text{ mm} \times 1.0\text{ mm}$ isotropic resolution via bilinear interpolation.
4. **Brain Foreground Cropping (`CropForegroundd`):** Crops bounding boxes strictly around non-zero brain tissue voxels, eliminating non-informative background air.
5. **Spatial Resizing (`Resized`):** Resizes the cropped volume to a standardized spatial cube of $96 \times 96 \times 96$ voxels via trilinear interpolation.
6. **Intensity Normalization (`NormalizeIntensityd`):** Applies channel-wise, non-zero z-score normalization:
   $$\tilde{x}_{c}(v) = \frac{x_{c}(v) - \mu_{c,\text{nonzero}}}{\sigma_{c,\text{nonzero}}}, \quad \forall v \in \{v \mid x_c(v) > 0\}$$
7. **Tensor Conversion (`EnsureTyped`, `ToTensord`):** Converts processed arrays into contiguous 32-bit floating-point PyTorch tensors.

### 2.4.2 Stochastic Training Augmentation Pipeline
During model training, the deterministic geometric pipeline was augmented with stochastic transformations to improve representation robustness:
- Random 3D spatial axis flips (`RandFlipd`) across axial, coronal, and sagittal axes ($p = 0.50$).
- Random 90-degree in-plane spatial rotations (`RandRotate90d`, $p = 0.50$).
- Random isotropic spatial zooming (`RandZoomd`, range $[0.9, 1.1]$, $p = 0.30$).
- Random intensity scaling (`RandScaleIntensityd`, factors $\pm 10\%$, $p = 0.50$).
- Random intensity shifting (`RandShiftIntensityd`, offsets $\pm 10\%$, $p = 0.50$).
- Random additive Gaussian noise (`RandGaussianNoised`, $\mu = 0.0, \sigma = 0.05$, $p = 0.20$).

## 2.5 Deep Learning Model Architectures
To evaluate cross-architectural generalizability, three canonical 3D deep learning architectures representing diverse inductive bias families were implemented:

1. **3D Residual Network (ResNet18 MedicalNet):** A 3D adaptation of the 18-layer residual network initialized with transfer learning weights pre-trained on aggregated multi-organ volumetric CT/MRI segmentation datasets (MedicalNet). The native fully connected layer was replaced with a custom classification head comprising a Dropout layer ($p = 0.40$) followed by a linear projection ($\mathbb{R}^{512} \to \mathbb{R}^{2}$).
2. **3D Densely Connected Network (DenseNet121 3D):** A 3D Densely Connected Convolutional Network with four dense blocks and transition layers, promoting iterative feature reuse and multi-scale gradient propagation. The final linear classifier was wrapped with Dropout ($p = 0.40$) mapping bottleneck features ($\mathbb{R}^{1024} \to \mathbb{R}^{2}$).
3. **3D Hierarchical Vision Transformer (SwinUNETR 3D):** A hierarchical vision transformer that utilizes shifted window self-attention mechanisms to capture long-range spatial dependencies. 3D patch tokens are processed across 4 resolution stages ($C=24$, feature expansion factors $[1, 2, 4, 8]$). Bottleneck feature representations ($\mathbb{R}^{384}$) were extracted via 3D adaptive average pooling, followed by Dropout ($p = 0.40$) and linear classification ($\mathbb{R}^{384} \to \mathbb{R}^{2}$).

## 2.6 Training Configuration and Hyperparameters
All neural network training procedures were executed under fixed, audited optimization constraints:
- **Optimizer:** AdamW with initial learning rate $\eta = 1 \times 10^{-4}$ and decoupled weight decay $\lambda = 1 \times 10^{-4}$.
- **Batch Size:** $B = 2$ volumes per mini-batch with gradient accumulation steps $= 1$.
- **Precision:** Automatic Mixed Precision (PyTorch AMP `float16` forward execution with gradient scaling).
- **Epoch Budget & Early Stopping:** Maximum 30 epochs with early stopping monitored on validation Macro-F1 (patience $= 6$ epochs, $\Delta_{\text{min}} = 0.0$, tolerance $\epsilon = 1 \times 10^{-4}$). Ties in validation Macro-F1 were broken deterministically by validation LGG recall.
- **Hardware Platform:** NVIDIA RTX A1000 GPU (8 GB dedicated GDDR6 VRAM) running CUDA 12.x on Windows 11 host.

## 2.7 Class-Imbalance Loss Protocols
To investigate the empirical effects of loss-level class re-weighting during training, three controlled loss protocols were formulated within standard Cross-Entropy:

$$\mathcal{L}_{\text{WCE}}(z, y) = - \sum_{k=0}^{K-1} w_k \cdot \mathbb{I}(y = k) \cdot \log\left(\frac{e^{z_k}}{\sum_{j=0}^{K-1} e^{z_j}}\right)$$

where $K = 2$, $N$ is total training samples, and $n_k$ is the class sample count.

- **Protocol A (Standard Unweighted Cross-Entropy):** Uniform class weighting:
  $$w_{\text{LGG}} = 1.0, \quad w_{\text{HGG}} = 1.0$$
- **Protocol B (Normalized Inverse-Frequency Weighting):** Inverse-frequency scaling rescaled such that $\sum_{k=0}^{K-1} w_k = K$:
  $$\tilde{w}_k = \frac{N}{K \cdot n_k}, \quad w_k = \tilde{w}_k \cdot \frac{K}{\sum_{j=0}^{K-1} \tilde{w}_j}$$
- **Protocol C (Adjusted Minority Emphasis):** Protocol B weights with the minority class (LGG) multiplied by $1.25$ and subsequently renormalized so that $\sum_{k=0}^{K-1} w_k = K$.

## 2.8 Focal Loss Formulation
To evaluate dynamic sample-hardness modulation without heuristic static weighting, Focal Loss was implemented and evaluated on the ResNet18 MedicalNet architecture:

$$\mathcal{L}_{\text{Focal}}(z, y) = - \alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

where $p_t = \text{softmax}(z)_y$ represents the model's estimated probability for the ground-truth class $y$, $\alpha_t = 1.0$ is the uniform class multiplier, and $\gamma = 2.0$ is the focusing parameter. Training was governed by identical early stopping constraints (terminating at epoch 14 with validation $\text{Macro-F1} = 0.7225$, validation LGG recall $= 50.0\%$).

## 2.9 Locked-Test Evaluation Protocol
Model checkpoints selected by the validation early stopping protocol were frozen and transferred to the read-only locked test cohort ($n = 188$). Evaluation was executed under `torch.no_grad()` and `model.eval()` mode with zero parameter updates.

### 2.9.1 Discrimination and Classification Metrics
- **Classification Accuracy:** $\text{Acc} = \frac{\text{TP} + \text{TN}}{N}$
- **Class-Specific Recalls:**
  $$\text{LGG Recall (Specificity)} = \frac{\text{TN}}{\text{TN} + \text{FP}}, \quad \text{HGG Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Macro-Averaged F1-Score:**
  $$\text{Macro-F1} = \frac{1}{2} \left( \text{F1}_{\text{LGG}} + \text{F1}_{\text{HGG}} \right)$$
- **Area Under ROC Curve (ROC-AUC):** Non-parametric Wilcoxon-Mann-Whitney ranking statistic:
  $$\text{ROC-AUC} = \frac{1}{n_{\text{LGG}} \cdot n_{\text{HGG}}} \sum_{i \in \text{HGG}} \sum_{j \in \text{LGG}} \mathbb{I}\left(P_i(\text{HGG}) > P_j(\text{HGG})\right) + \frac{1}{2}\mathbb{I}\left(P_i = P_j\right)$$

### 2.9.2 Calibration Metrics
- **Expected Calibration Error (ECE):** Top-class confidence predictions grouped into $M = 10$ equally spaced probability bins $B_m \subset (0, 1]$:
  $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Brier Score:** Mean squared error between predicted HGG posterior risk and binary ground truth ($y \in \{0, 1\}$):
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} \left( P_i(\text{HGG}) - y_i \right)^2$$
- **Negative Log-Likelihood (NLL):** Multi-class cross-entropy loss evaluated on test probabilities:
  $$\text{NLL} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{k=0}^{K-1} y_{ik} \log\left(P_i(k)\right)$$

## 2.10 Decision Threshold Operating-Point Analysis
To evaluate continuous sensitivity/specificity trade-offs, predicted class probabilities $P(\text{LGG}) = 1.0 - P(\text{HGG})$ were converted to binary diagnostic classifications via parameterized decision cutoffs:

$$\hat{y}(\tau) = \begin{cases} 0 \quad (\text{LGG}), & \text{if } P(\text{LGG}) \geq \tau_{\text{LGG}} \\ 1 \quad (\text{HGG}), & \text{otherwise} \end{cases}$$

Threshold sweeps were evaluated across $\tau \in [0.10, 0.90]$ in increments of $0.05$. 

*Methodological Note:* All test-set threshold sweeps are reported as **descriptive operating-characteristic analyses** to map empirical sensitivity potential across the probability domain; they are explicitly distinguished from prospective threshold selection protocols intended for clinical deployment.

## 2.11 Post-Hoc Temperature Scaling
To decompress distorted probability distributions without altering learned discriminative feature representations, post-hoc temperature scaling was applied.

1. **Validation Optimization:** Raw output logits $z_i \in \mathbb{R}^2$ from the 187-subject validation cohort were collected. A single scalar parameter $T > 0$ was optimized strictly by minimizing validation Negative Log-Likelihood via bounded scalar minimization (`scipy.optimize.minimize_scalar`, bounds $[0.01, 10.0]$):
   $$T^* = \arg\min_{T > 0} - \sum_{i=1}^{N_{\text{val}}} \log\left( \frac{e^{z_{i, y_i} / T}}{\sum_{j=0}^{1} e^{z_{i, j} / T}} \right)$$
2. **Parameter Freezing:** The optimal temperature scalar $T^*$ was frozen ($T^* = 0.4923$ for Focal ResNet18). Zero test-set labels or test logits were exposed during parameter estimation.
3. **Locked-Test Evaluation:** Calibrated probabilities for locked-test subjects were computed by dividing frozen test logits by $T^*$:
   $$P_{\text{cal}}(k \mid x) = \frac{e^{z_k(x) / T^*}}{\sum_{j=0}^{1} e^{z_j(x) / T^*}}$$

*Mathematical Invariance Property:* Because scalar division by $T^* > 0$ is a strictly monotonic logit transformation, the relative ranking of all subjects is mathematically invariant, ensuring $\Delta \text{ROC-AUC} = 0.0000000000$ and guaranteeing $100\%$ prediction equivalence at the default decision boundary ($\tau = 0.50$).

## 2.12 Cross-Architecture Calibration Benchmark
To evaluate whether temperature scaling generalizes across diverse deep learning architectures and loss formulations, post-hoc temperature scaling was benchmarked across all nine canonical baseline checkpoints (ResNet18, DenseNet121, SwinUNETR under Protocols A, B, and C) alongside the Focal Loss reference model. For each of the 10 models, validation logits were extracted, $T^*$ was optimized strictly on validation NLL, and calibrated calibration metrics (ECE, Brier, NLL) were evaluated on the locked test cohort.

## 2.13 Predictive Uncertainty Profiling
Sample-level predictive uncertainty was quantified via information-theoretic predictive Shannon entropy computed across output probabilities:

$$H(P) = - \sum_{k=0}^{1} P_k \cdot \log_2(P_k)$$

where $H(P) \in [0.0, 1.0]\text{ bits}$. Cases situated near the decision boundary ($P(\text{HGG}) \approx 0.50$) exhibit maximal predictive entropy ($H \to 1.0\text{ bit}$), providing a quantitative flag for identifying ambiguous or borderline cases requiring secondary radiological review.

## 2.14 Statistical and Evaluation Framework
To prevent distorted conclusions arising from class imbalance, the evaluation framework enforces four reporting rules:
1. Aggregate accuracy is never reported in isolation; it must be accompanied by Macro-F1 and class-stratified recalls ($\text{LGG Recall}$ and $\text{HGG Recall}$).
2. Ranking discrimination ($\text{ROC-AUC}$) is reported separately from threshold-dependent operational metrics.
3. Probability calibration ($\text{ECE}$ and Brier score) is evaluated to measure statistical confidence reliability independently of classification accuracy.
4. Descriptive threshold sweeps are presented as continuous operating manifolds, distinguishing representation capability from arbitrary cutoff selection.

## 2.15 Reproducibility Controls and Zero-Leakage Architecture
To ensure complete experimental auditability and prevent data leakage:
- Deterministic random seeding (`seed = 42`) was enforced across PyTorch, NumPy, and CUDA environments.
- Checkpoint integrity was documented using SHA-256 cryptographic hashes for the archived checkpoint set covered by the integrity manifest, while other finalized checkpoints were preserved in their designated experiment directories.
- Training, validation, and test cohorts were permanently segregated via read-only manifest files.
- Calibration temperature parameters ($T^*$) were optimized exclusively on validation data.
- The locked test cohort was evaluated strictly in forward read-only mode without model selection or iterative threshold tuning.
