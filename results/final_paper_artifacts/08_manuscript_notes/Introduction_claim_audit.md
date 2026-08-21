# Introduction Section Scientific Claim-Evidence Audit

This audit evaluates the scientific strength, potential overclaims, and adopted conservative framing for all major assertions in the Introduction section.

| Paragraph | Claim Topic | Evidence Strength | Potential Overclaim | Adopted Conservative Wording |
| --- | --- | --- | --- | --- |
| Para 1 (Clinical context) | Asymmetric clinical risk in medical AI | CONCEPTUAL / LITERATURE | Claiming current AI models are ready for autonomous clinical deployment. | Framed as a diagnostic requirement for reliable decision support; no autonomous deployment claims made. |
| Para 2 (Class imbalance) | Minority category suppression in BraTS GLI | DEMONSTRATED | Claiming all imbalanced medical datasets behave identically. | Presented BraTS 2023 GLI grading (>90% HGG vs <10% LGG) as a representative clinical motivating setting. |
| Para 3 (Evaluation metrics) | ROC-AUC and Accuracy answering different questions | DEMONSTRATED | Declaring ROC-AUC invalid or useless for medical imaging. | Stated that ROC-AUC evaluates cohort-wide ranking discrimination, which does not determine sensitivity at a particular operating cutoff. |
| Para 4 (Loss interventions) | Loss weighting altering optimization and posterior distributions | DEMONSTRATED (Empirical) | Claiming class weighting always fails or causally proves architectural inductive bias. | Stated that loss interventions can alter optimization and posterior confidence in complex, architecture-dependent ways. |
| Para 5 (Calibration & Uncertainty) | Calibration distinct from sensitivity and operating thresholds | DEMONSTRATED | Claiming temperature scaling solves the minority class classification problem. | Stated that calibration restores statistical fidelity of posterior probabilities, but does not dictate whether default thresholds meet sensitivity requirements. |
| Para 6 (Research gap) | Integrated multidimensional evaluation gap under severe imbalance | LITERATURE GAP | Claiming nobody has ever studied calibration or loss weighting before. | Framed as an evaluation integration gap: examining the interaction of loss x architecture x calibration x thresholds under severe class imbalance. |
| Para 7 (Contributions) | 4-point contribution statement with locked-test controls | DEMONSTRATED (Experimental Chain) | Claiming a novel loss function or a clinically validated diagnostic system. | Positioned as a controlled empirical evaluation and multidimensional evaluation framework on a locked test benchmark. |
