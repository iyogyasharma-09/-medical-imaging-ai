# 2. Related Work

## 2.1 Deep Learning for Medical Image Classification
Deep learning has transformed medical image analysis, with convolutional neural networks (CNNs) achieving radiologist-level performance across diverse diagnostic tasks including dermatology, retinal imaging, pathology, and neuroradiology [1]. The foundational survey by Litjens et al. [1] documented over 300 applications across classification, detection, and segmentation, establishing deep learning as the dominant computational paradigm in medical imaging. Within volumetric neuroimaging, 3D convolutional architectures have become standard for exploiting spatial context in multi-parametric MRI [2, 3].

Residual learning [2] introduced skip connections that enabled training of substantially deeper networks, and dense connectivity [3] further improved feature reuse and parameter efficiency. Transfer learning from large-scale pre-trained models has accelerated convergence in data-limited medical domains; MedicalNet [4] specifically provided 3D ResNet weights pre-trained on aggregated medical segmentation challenges, enabling effective initialization for volumetric tasks. More recently, vision transformers [5] have been adapted to 3D medical imaging through architectures such as SwinUNETR [6], which applies hierarchical shifted-window self-attention to brain tumor analysis.

Despite their discriminative capacity, the evaluation of these models in clinical contexts has predominantly relied on aggregate metrics such as overall accuracy and area under the receiver operating characteristic curve (ROC-AUC). While these measures capture important aspects of model performance, they may not reveal critical class-specific failure modes when diagnostic categories exhibit highly skewed prevalence distributions [7, 8].

## 2.2 Glioma Classification and the BraTS Benchmark
Diffuse gliomas are the most common primary malignant brain tumors in adults, and their histopathological grading carries direct prognostic and therapeutic implications. The Brain Tumor Segmentation (BraTS) challenge, established by Menze et al. [9] and expanded by Bakas et al. [10], has provided standardized multi-institutional, multi-parametric MRI datasets spanning T1-weighted, contrast-enhanced T1, T2-weighted, and FLAIR sequences. The BraTS 2021 benchmark [11] further extended the challenge to include radiogenomic classification alongside segmentation, and subsequent iterations including BraTS 2023 continued to refine data curation and task definitions for adult glioma analysis.

Within this benchmark, grading low-grade gliomas (LGG, WHO grade 2/3) from high-grade glioblastoma (HGG, WHO grade 4) represents a clinically meaningful binary classification task. However, the natural prevalence of glioblastoma substantially exceeds that of lower-grade tumors in clinical cohorts, resulting in severe class imbalance that complicates model training and evaluation. While prior BraTS studies have extensively addressed tumor segmentation, fewer investigations have systematically evaluated the reliability of classification models across discrimination, calibration, and class-stratified sensitivity within this benchmark.

## 2.3 Class Imbalance in Medical Imaging
Class imbalance is pervasive in clinical imaging, where pathological findings of interest often represent a small fraction of the total cohort [12, 13]. Buda et al. [12] conducted a systematic study of class imbalance in CNNs across MNIST, CIFAR-10, and ImageNet, demonstrating that imbalanced training distributions detrimentally affect classification performance and that oversampling was the most consistently effective mitigation. Johnson and Khoshgoftaar [13] surveyed deep learning approaches to class imbalance, categorizing solutions into data-level methods (oversampling, undersampling, synthetic generation), algorithm-level methods (cost-sensitive learning, class-weighted loss functions), and hybrid strategies.

In medical imaging specifically, class weighting through inverse-frequency or manually adjusted loss scaling is frequently applied as a straightforward training-time intervention. However, prior work has primarily evaluated these interventions through aggregate metrics such as overall accuracy or weighted F1-score, rather than through systematic class-stratified sensitivity analysis. The distinction between aggregate performance improvement and minority-class diagnostic sensitivity is critical in clinical grading tasks where the cost of missing a rare but aggressive pathology substantially exceeds the cost of a false positive.

## 2.4 Focal Loss and Hardness-Aware Optimization
Focal Loss, introduced by Lin et al. [14] for dense object detection, modulates the standard cross-entropy loss by a factor $(1 - p_t)^\gamma$, down-weighting the contribution of well-classified (easy) examples and concentrating optimization on hard, misclassified samples. The original formulation was motivated by extreme foreground-background imbalance in object detection, and the technique has since been widely adopted in medical imaging applications including retinal disease classification, chest radiograph interpretation, and histopathology analysis.

Mukhoti et al. [15] subsequently investigated the calibration properties of Focal Loss, demonstrating that appropriate selection of the focusing parameter $\gamma$ can yield well-calibrated predictions in certain settings. However, they also observed that Focal Loss can produce under-confident predictions when $\gamma$ is set too high, effectively compressing the output probability distribution. This observation suggests that the effect of Focal Loss on posterior probability calibration may require separate evaluation from its effect on discriminative ranking, a distinction that has received limited systematic attention in the 3D medical imaging classification literature.

## 2.5 Probability Calibration in Deep Learning
Probability calibration—the alignment between predicted confidence scores and empirical event frequencies—has emerged as a critical quality dimension for deep learning models deployed in risk-sensitive domains. Guo et al. [16] demonstrated that modern deep neural networks, despite achieving high classification accuracy, are frequently miscalibrated, producing overconfident predictions that do not reflect true class likelihoods. They proposed reliability diagrams and Expected Calibration Error (ECE) [17] as standard evaluation tools, alongside the Brier score as a proper scoring rule that jointly captures discrimination and calibration.

Subsequent work by Minderer et al. [18] revisited calibration in the context of larger and more modern architectures, observing that calibration behavior varies with model family, scale, and training recipe. In medical imaging, Mehrtash et al. [19] specifically evaluated confidence calibration and predictive uncertainty estimation for deep medical image segmentation, highlighting the clinical relevance of well-calibrated probability outputs for downstream clinical decision-making. Despite this growing body of work, systematic evaluation of how calibration interacts with class-stratified sensitivity and decision threshold selection under severe class imbalance remains limited.

## 2.6 Post-Hoc Calibration and Temperature Scaling
Post-hoc calibration methods adjust model outputs after training to improve probability reliability without modifying learned feature representations. Platt scaling [20] introduced logistic regression on model outputs as a calibration technique for support vector machines. Guo et al. [16] demonstrated that a simplified variant—temperature scaling—which divides logits by a single learned scalar parameter $T$ before softmax, is remarkably effective for modern deep networks while preserving the computational simplicity of a single-parameter optimization.

Crucially, temperature scaling is a strictly monotonic transformation of the logit space: dividing all logits by a positive scalar $T > 0$ preserves the relative ranking of predictions, leaving ROC-AUC and the default $\tau = 0.50$ decision boundary mathematically invariant. The parameter $T$ is optimized by minimizing negative log-likelihood (NLL) on a held-out validation set and subsequently frozen before application to test data. This property makes temperature scaling particularly suitable for clinical evaluation protocols that require strict separation between calibration fitting and test-set evaluation. However, it is important to recognize that temperature scaling modifies probability reliability but does not alter discriminative ranking or automatically adjust operational decision thresholds to meet clinical sensitivity requirements.

## 2.7 Uncertainty Estimation and Reliable Medical AI
Quantifying predictive uncertainty is increasingly recognized as essential for trustworthy medical AI systems [21, 22]. Kendall and Gal [23] formalized the distinction between aleatoric uncertainty (irreducible noise inherent in the data) and epistemic uncertainty (reducible uncertainty due to limited model knowledge), proposing a unified Bayesian deep learning framework. Gal and Ghahramani [24] demonstrated that standard dropout training can be reinterpreted as approximate Bayesian inference (Monte Carlo Dropout), enabling practical uncertainty estimation without architectural modifications.

Lakshminarayanan et al. [25] proposed deep ensembles as a scalable alternative, demonstrating strong uncertainty calibration through aggregation of independently trained models. In the medical imaging context, Jungo et al. [26] evaluated uncertainty estimation methods for brain tumor segmentation, finding that while many approaches are well-calibrated at the dataset level, subject-level miscalibration remains a significant challenge. Kompa et al. [21] argued that communicating uncertainty is essential for clinical adoption, framing uncertainty quantification as a prerequisite for responsible AI deployment rather than an optional diagnostic supplement.

Despite these advances, uncertainty estimation is most commonly evaluated independently of calibration, class-stratified sensitivity, and decision threshold analysis. Few studies have examined how predictive entropy relates to minority-class misclassification patterns under severe class imbalance, or how uncertainty information complements post-hoc calibration in guiding operational threshold selection.

## 2.8 Evaluation Gap and Positioning of the Present Study
The preceding literature establishes that deep learning for medical image classification [1], class imbalance mitigation [12, 13, 14], probability calibration [16, 18, 19], post-hoc temperature scaling [16, 20], decision threshold analysis, and predictive uncertainty estimation [21, 23, 24, 25] are each individually well-studied research areas. However, these dimensions are predominantly evaluated in isolation: classification studies report aggregate accuracy and AUC without systematic calibration assessment; calibration studies evaluate ECE and reliability diagrams without class-stratified sensitivity analysis; and uncertainty studies quantify predictive entropy without connecting it to operational threshold selection under clinical class imbalance.

The critical gap that the present study addresses is not the novelty of any individual component, but rather the absence of a controlled, integrated evaluation protocol that examines how these dimensions interact under severe class imbalance in a fixed 3D medical imaging setting. Specifically, we investigate whether ranking discrimination, probability calibration, class-specific sensitivity, decision-threshold behavior, and predictive uncertainty provide complementary or redundant evaluative information when models are trained, calibrated, and evaluated under identical data conditions across multiple architectures and loss formulations. By systematically disaggregating these dimensions on the BraTS 2023 glioma grading benchmark, we aim to demonstrate that trustworthy clinical evaluation requires multidimensional assessment rather than reliance on any single aggregate metric.

---

## References

[1] Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60–88. DOI: 10.1016/j.media.2017.07.005

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proc. IEEE CVPR*, 770–778. DOI: 10.1109/CVPR.2016.90

[3] Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. In *Proc. IEEE CVPR*. DOI: 10.1109/CVPR.2017.243

[4] Chen, S., Ma, K., & Zheng, Y. (2019). Med3D: Transfer learning for 3D medical image analysis. *arXiv preprint arXiv:1904.00625*.

[5] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In *Proc. ICLR 2021*. DOI: 10.48550/arXiv.2010.11929

[6] Hatamizadeh, A., Nath, V., Tang, Y., et al. (2022). Swin UNETR: Swin Transformers for semantic segmentation of brain tumors in MRI images. In *BrainLes 2021, LNCS*. DOI: 10.1007/978-3-031-08999-2_22

[7] Hicks, S. A., Strümke, I., Thambawita, V., et al. (2022). On evaluation metrics for medical applications of artificial intelligence. *Scientific Reports*, 12, 5979. DOI: 10.1038/s41598-022-09954-8

[8] Varoquaux, G. & Cheplygina, V. (2022). Machine learning for medical imaging: methodological failures and recommendations for the future. *npj Digital Medicine*, 5, 48. DOI: 10.1038/s41746-022-00592-y

[9] Menze, B. H., Jakab, A., Bauer, S., et al. (2015). The multimodal brain tumor image segmentation benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024. DOI: 10.1109/TMI.2014.2377694

[10] Bakas, S., Akbari, H., Sotiras, A., et al. (2017). Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. *Scientific Data*, 4, 170117. DOI: 10.1038/sdata.2017.117

[11] Baid, U., Ghodasara, S., Mohan, S., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. *arXiv preprint arXiv:2107.02314*. DOI: 10.48550/arXiv.2107.02314

[12] Buda, M., Maki, A., & Mazurowski, M. A. (2018). A systematic study of the class imbalance problem in convolutional neural networks. *Neural Networks*, 106, 249–259. DOI: 10.1016/j.neunet.2018.07.011

[13] Johnson, J. M. & Khoshgoftaar, T. M. (2019). Survey on deep learning with class imbalance. *Journal of Big Data*, 6, 27. DOI: 10.1186/s40537-019-0192-5

[14] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proc. IEEE ICCV*, 2980–2988. DOI: 10.1109/ICCV.2017.324

[15] Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P. H. S., & Dokania, P. K. (2020). Calibrating deep neural networks using focal loss. *Advances in Neural Information Processing Systems*, 33, 15288–15299.

[16] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proc. ICML 2017*. DOI: 10.48550/arXiv.1706.04599

[17] Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning into quantiles. In *Proc. AAAI 2015*. DOI: 10.1609/aaai.v29i1.9602

[18] Minderer, M., Djolonga, J., Romber, R., et al. (2021). Revisiting the calibration of modern neural networks. *Advances in Neural Information Processing Systems*, 34, 15682–15694. DOI: 10.48550/arXiv.2106.07998

[19] Mehrtash, A., Wells, W. M., Tempany, C. M., Abolmaesumi, P., & Kapur, T. (2020). Confidence calibration and predictive uncertainty estimation for deep medical image segmentation. *IEEE Transactions on Medical Imaging*, 39(12), 3868–3878. DOI: 10.1109/TMI.2020.3006437

[20] Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*, MIT Press, 61–74.

[21] Kompa, B., Snoek, J., & Beam, A. L. (2021). Second opinion needed: communicating uncertainty in medical machine learning. *npj Digital Medicine*, 4, 4. DOI: 10.1038/s41746-020-00367-3

[22] Pfohl, S. R., Duan, T., Ding, D. Y., & Shah, N. H. (2022). Net benefit, calibration, threshold selection, and training objectives for algorithmic fairness in healthcare. In *Proc. ACM FAccT 2022*. DOI: 10.1145/3531146.3533221

[23] Kendall, A. & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *Advances in Neural Information Processing Systems*, 30. DOI: 10.48550/arXiv.1703.04977

[24] Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *Proc. ICML 2016*. DOI: 10.48550/arXiv.1506.02142

[25] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems*, 30. DOI: 10.48550/arXiv.1612.01474

[26] Jungo, A., Balsiger, F., & Reyes, M. (2020). Analyzing the quality and challenges of uncertainty estimations for brain tumor segmentation. *Frontiers in Neuroscience*, 14, 282. DOI: 10.3389/fnins.2020.00282
