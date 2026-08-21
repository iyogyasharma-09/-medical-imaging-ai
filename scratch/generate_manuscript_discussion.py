"""
generate_manuscript_discussion.py
----------------------------------
Generates the publication-ready Discussion section markdown draft and the
Discussion claim audit markdown file under results/final_paper_artifacts/08_manuscript_notes/.

Strict Rules:
- Conservative, rigorous scientific interpretation.
- No causal overreach or promotional claims.
- Disaggregates discrimination, calibration, thresholds, and architecture responses.
- Clear separation of methodological findings from clinical deployment recommendations.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\mri\mri-")
NOTES_DIR = BASE_DIR / "results" / "final_paper_artifacts" / "08_manuscript_notes"
NOTES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Discussion Section Draft
# ---------------------------------------------------------------------------
discussion_draft_text = """# 4. Discussion

## 4.1 Principal Findings
In this study, we evaluated the multidimensional reliability of 3D deep learning models for medical image classification under severe class imbalance, using the BraTS 2023 glioma grading benchmark ($93.6\\%$ HGG vs $6.4\\%$ LGG in the locked test cohort). Our primary empirical finding is that four critical dimensions of model performance—ranking discrimination, probability calibration, class-stratified sensitivity, and operational decision thresholding—frequently decouple under extreme class imbalance. 

Specifically, models demonstrating high ranking capacity (e.g., SwinUNETR under standard Cross-Entropy achieving $\\text{ROC-AUC} = 0.9330$) exhibited complete minority-class diagnostic suppression at the standard decision boundary ($0.00\\%$ LGG recall at $\\tau = 0.50$). Conversely, loss-level interventions such as Focal Loss maintained strong ranking discrimination ($\\text{ROC-AUC} = 0.9541$), but were associated with substantial probability scale compression (uncalibrated $\\text{ECE} = 9.75\\%$) that constrained minority sensitivity at default cutoffs ($41.67\\%$ LGG recall). Validation-only post-hoc temperature scaling successfully repaired probability calibration ($65.0\\%$ test ECE reduction to $3.41\\%$) without altering ranking discrimination, enabling principled exploration of continuous sensitivity operating points on the calibrated probability scale. Together, these results provide empirical evidence that evaluating medical image classifiers by a single aggregate metric can obscure critical reliability failures, motivating an integrated, multidimensional evaluation framework.

## 4.2 Why Aggregate Performance Metrics Are Insufficient
Standard clinical machine learning studies predominantly report aggregate classification accuracy and area under the receiver operating characteristic curve (ROC-AUC). Our empirical findings demonstrate that neither metric alone provides sufficient information regarding minority-class clinical reliability in imbalanced diagnostic tasks.

Aggregate accuracy is fundamentally dominated by majority-class volume. When the minority pathology represents only $6.4\\%$ of the evaluated cohort, a trivial majority-class classifier attains $93.62\\%$ accuracy while missing $100\\%$ of minority cases. Correspondingly, SwinUNETR under Protocol A achieved $93.62\\%$ overall accuracy despite diagnosing zero low-grade gliomas ($0/12$). 

Similarly, while ROC-AUC is prevalence-invariant and measures the probability that a randomly chosen positive case is ranked above a randomly chosen negative case, high ROC-AUC does not determine the sensitivity achieved at a particular operational decision threshold. A model with an ROC-AUC above $0.93$ may place all minority predictions in a low absolute probability range (e.g., $P \\in [0.02, 0.08]$), yielding zero sensitivity under standard decision rules ($\tau = 0.50$). Thus, while ROC-AUC answers a cohort ranking question and accuracy answers an overall sample-weighted question, neither metric assesses whether class-stratified sensitivity meets task-specific clinical requirements.

## 4.3 Separation of Discrimination, Calibration, and Decision Threshold
A central conceptual contribution supported by our experimental sequence is the formal separation of three evaluative dimensions:

1. **Discrimination (Ranking Capacity):** Governed by the separability of the learned feature representations. It is quantified by rank-based metrics such as ROC-AUC and remains mathematically invariant to positive scalar temperature transformations and decision threshold selection.
2. **Calibration (Statistical Reliability):** Governed by the alignment between predicted probabilities and empirical event frequencies across confidence bins. It is quantified by ECE and Brier score and can be adjusted post-hoc via monotonic logit scaling without retraining.
3. **Decision Threshold (Operational Boundary):** Governed by clinical utility, risk tolerance, and the trade-off between false negatives and false positives. It dictates how continuous probabilities are translated into discrete diagnostic actions.

Our Focal Loss experiment on ResNet18 empirically proves the independence of these dimensions:
- The raw model possessed high discrimination ($\\text{ROC-AUC} = 0.9541$), but poor calibration ($\\text{ECE} = 9.75\\%$) and modest default sensitivity ($41.67\\%$ LGG recall at $\\tau = 0.50$).
- Post-hoc temperature scaling ($T^* = 0.4923$) improved calibration ($\text{ECE} \\to 3.41\\%$, $\\text{Brier} \\to 0.0394$), yet left ROC-AUC ($0.9541$) and default classifications at $\\tau = 0.50$ ($41.67\\%$ recall) $100\\%$ identical.
- Subsequent decision threshold adjustment on the calibrated scale shifted operating sensitivity from $41.67\\%$ ($\tau = 0.50$) to $66.67\\%$ ($\tau = 0.35$), $83.33\\%$ ($\tau = 0.20$), and $100.00\\%$ ($\tau = 0.10$), while ROC-AUC remained strictly invariant.

These findings demonstrate that calibration is distinct from discrimination, and calibration alone does not automatically resolve threshold-dependent sensitivity limitations.

## 4.4 Implications of Class-Imbalance Interventions
In medical image analysis, loss-level re-weighting (e.g., inverse-frequency weighting or minority up-weighting) is frequently applied as a standard remedy for class imbalance. Our empirical evaluation across three architectures under Protocols B and C indicates that loss weighting is not a universally reliable solution.

Identical weighting protocols produced divergent responses across network backbones:
- For SwinUNETR, minority weighting under Protocol C mitigated default minority collapse, increasing LGG recall from $0.00\\%$ to $25.00\\%$ ($3/12$) and Macro-F1 from $0.4835$ to $0.6736$.
- Conversely, for ResNet18 and DenseNet121, class weighting systematically degraded minority sensitivity: ResNet18 LGG recall dropped from $75.00\\%$ (Protocol A) to $66.67\\%$ (Protocol B) and $41.67\\%$ (Protocol C), while DenseNet121 dropped from $66.67\\%$ (Protocol A) to $58.33\\%$ (Protocol B) and $33.33\\%$ (Protocol C).

These results demonstrate that class re-weighting does not provide a guaranteed or uniform sensitivity improvement. The mechanism governing these architecture-weighting interactions was not directly investigated in this study, but our empirical observations indicate that loss-weighting interventions should be evaluated with architecture-specific caution rather than assumed to be universally beneficial.

## 4.5 Focal Loss and Post-Hoc Calibration Dynamics
Focal Loss is widely employed to address class imbalance by dynamically modulating the loss contribution of well-classified majority samples. In our evaluation on ResNet18 MedicalNet, Focal Loss achieved strong discriminative feature learning ($\\text{ROC-AUC} = 0.9541$), but did not automatically resolve the minority sensitivity problem at the standard cutoff ($41.67\\%$ LGG recall at $\\tau = 0.50$).

Crucially, our analysis indicates that this behavior was associated with probability scale compression: modulating easy examples during training attenuated output logit magnitudes, concentrating predicted probabilities into a narrow sub-unity range and yielding an elevated test ECE of $9.75\\%$. When post-hoc temperature scaling was applied ($T^* = 0.4923$), the fitted parameter below unity confirmed that sharpening the logit distribution improved validation log-likelihood and resolved test-set probability compression, reducing test ECE to $3.41\\%$ and Brier score to $0.0394$.

Importantly, post-hoc calibration did not 'solve' the underlying classification task or alter default $\\tau = 0.50$ binary predictions; rather, it restored the statistical fidelity of posterior risk scores, ensuring that subsequent decision threshold selections operated on well-calibrated probabilities.

## 4.6 Architecture-Dependent Reliability Profiles
Our cross-architecture benchmark across ResNet18, DenseNet121, and SwinUNETR revealed distinct baseline reliability profiles under identical dataset splits and training protocols. 

Under standard Cross-Entropy, convolutional backbones (ResNet18 and DenseNet121) preserved moderate minority-class sensitivity at default cutoff ($75.00\\%$ and $66.67\\%$ recall), whereas the hierarchical vision transformer (SwinUNETR) suppressed minority predictions entirely ($0.00\\%$ recall) despite achieving comparable ranking discrimination ($\\text{ROC-AUC} = 0.9330$). Conversely, when minority weighting was introduced (Protocol C), SwinUNETR improved its sensitivity ($\text{AUC} = 0.9735$), while ResNet18 and DenseNet121 experienced sensitivity degradation.

Furthermore, our calibration benchmark (Table 3) revealed that good calibration does not imply high minority sensitivity: SwinUNETR Protocol A achieved a calibrated ECE of $1.12\\%$, yet diagnosed zero minority cases at default threshold. These observations suggest that model reliability is an emergent property resulting from the interaction between data distribution, network architecture, loss formulation, and operational decision boundaries, rather than a fixed attribute of a single model family.

## 4.7 Practical Implications for Clinical AI Evaluation
Based on our empirical findings, we propose the following methodological recommendations for evaluating medical image classifiers in imbalanced clinical domains:

1. **Mandatory Class-Stratified Reporting:** Always report class-specific sensitivity and specificity alongside aggregate accuracy and Macro-F1.
2. **Disaggregate Discrimination from Decision Cutoffs:** Report rank discrimination (ROC-AUC) as an index of representation quality, while treating decision threshold placement as a distinct operational choice.
3. **Assess Probability Calibration:** Routinely evaluate probability reliability using ECE, Brier score, and reliability diagrams; apply validation-only post-hoc calibration when utilizing margin-based or focal loss functions.
4. **Prospective Decision Threshold Selection:** Pre-specify operational decision thresholds on independent validation cohorts based on task-specific clinical utility functions (e.g., balancing the cost of missed diagnoses against false positive biopsies), rather than assuming $\\tau = 0.50$ is optimal.
5. **Preserve Locked Test Cohorts:** Maintain strict test-set isolation during calibration fitting and threshold selection to prevent over-optimistic performance estimation.
6. **Incorporate Predictive Uncertainty:** Utilize predictive entropy to flag borderline and high-uncertainty cases for secondary radiological review.

These recommendations provide a structured framework to improve the transparency and trustworthiness of deep learning evaluations prior to clinical consideration.

## 4.8 Limitations
The conclusions of this study should be interpreted within the context of several important methodological limitations:

1. **Limited Minority Cohort Size:** The locked test cohort contained $12$ low-grade glioma subjects ($6.4\\%$ prevalence), reflecting the clinical distribution of the BraTS benchmark but limiting the statistical granularity for higher-order subgroup stratification.
2. **Single-Seed Canonical Training:** Models were evaluated on fixed, reproducible random seeds; systematic multi-seed optimization trajectory variance was not exhaustively isolated across all combinations.
3. **Descriptive Nature of Test-Set Threshold Sweeps:** The reported operating points on the locked test cohort are descriptive demonstrations of sensitivity trade-offs; operational thresholds for deployment require prospective validation under pre-specified utility functions.
4. **Mechanisms of Architectural Bias Not Isolated:** The underlying mathematical and layer-wise causes of divergent weighting responses between CNNs and Vision Transformers were not causally investigated.
5. **Absence of Patient-Specific Confounders:** Voxel-level morphometric factors (e.g., total lesion volume, necrosis proportion, anatomical location) were not modeled as individual covariates.
6. **Single-Parameter Calibration:** Post-hoc calibration was evaluated using scalar temperature scaling; multi-parameter vector scaling and non-parametric spline calibration were not investigated.
7. **Retrospective Evaluation:** The evaluation was conducted retrospectively on standardized benchmark data; prospective clinical deployment trials across multiple scanner vendors were outside the study scope.

## 4.9 Future Directions
Several promising research avenues emerge from this work:

1. **Multi-Seed Variance Isolation:** Systematically characterizing stochastic optimizer variance across diverse architectures under extreme class imbalance.
2. **Prospective Clinical Utility Optimization:** Formulating explicit cost-matrix optimization protocols on validation splits to select task-specific operating thresholds prior to deployment.
3. **Morphometric Uncertainty Modeling:** Integrating voxel-level lesion morphology and anatomical priors directly into predictive uncertainty estimations.
4. **Multi-Center External Validation:** Evaluating the generalizability of the 5-pillar framework across distinct clinical scanner vendors, field strengths, and acquisition protocols.
5. **Investigation of Inductive Bias Dynamics:** Conducting mechanistic loss-landscape analyses to understand why Vision Transformers and CNNs exhibit divergent gradient dynamics under loss re-weighting.
"""

(NOTES_DIR / "Discussion_section_draft.md").write_text(discussion_draft_text)

# ---------------------------------------------------------------------------
# 2. Discussion Claim Audit
# ---------------------------------------------------------------------------
claim_audit_records = [
    {
        "Claim_Topic": "4.1 Decoupling of evaluation dimensions",
        "Empirical_Evidence": "SwinUNETR Protocol A achieves 0.9330 AUC with 0.0% LGG recall; Focal ResNet18 achieves 0.9541 AUC with 9.75% ECE repaired to 3.41% without changing AUC.",
        "Evidence_Strength": "DEMONSTRATED",
        "Potential_Overclaim": "Claiming that all medical models decouple in this manner universally.",
        "Recommended_Wording": "State that discrimination, calibration, class-stratified sensitivity, and decision thresholds frequently decouple under severe class imbalance, as demonstrated on BraTS 2023.",
    },
    {
        "Claim_Topic": "4.2 Insufficiency of ROC-AUC and Accuracy",
        "Empirical_Evidence": "93.62% accuracy achieved with 0% minority recall; 0.9330 AUC achieved with majority-only default predictions.",
        "Evidence_Strength": "DEMONSTRATED",
        "Potential_Overclaim": "Claiming ROC-AUC is useless or flawed.",
        "Recommended_Wording": "State that ROC-AUC answers a ranking question and accuracy answers an overall sample-weighted question; neither determines the sensitivity achieved at a specific operating threshold.",
    },
    {
        "Claim_Topic": "4.3 Mathematical independence of calibration & ranking",
        "Empirical_Evidence": "Temperature scaling T*=0.4923 reduced test ECE by 65.0% while test ROC-AUC (0.9541) and tau=0.50 predictions remained strictly invariant.",
        "Evidence_Strength": "DEMONSTRATED",
        "Potential_Overclaim": "Claiming calibration improves classification performance.",
        "Recommended_Wording": "State that calibration modifies the statistical reliability of probabilities, but mathematically preserves rank discrimination and default binary classifications.",
    },
    {
        "Claim_Topic": "4.4 Architecture-dependent class weighting response",
        "Empirical_Evidence": "SwinUNETR improved under Protocol C (0% -> 25% recall), while ResNet18 degraded (75% -> 41.7% recall) and DenseNet121 degraded (66.7% -> 33.3% recall).",
        "Evidence_Strength": "DEMONSTRATED (Empirical)",
        "Potential_Overclaim": "Claiming that class weighting proves architectural inductive bias causally.",
        "Recommended_Wording": "State that models exhibited divergent empirical responses to identical weighting protocols, and explicitly state that the underlying causal mechanism was not directly investigated.",
    },
    {
        "Claim_Topic": "4.5 Focal Loss probability compression",
        "Empirical_Evidence": "Focal ResNet18 had raw ECE=9.75% with T*=0.4923, reducing ECE to 3.41% on locked test cohort.",
        "Evidence_Strength": "SUPPORTED (Empirical)",
        "Potential_Overclaim": "Claiming Focal Loss universally causes probability compression in all datasets.",
        "Recommended_Wording": "State that Focal Loss in this experiment was associated with compressed probability outputs, and the fitted temperature below unity indicated that sharpening logits improved validation likelihood and test calibration.",
    },
    {
        "Claim_Topic": "4.6 Architecture reliability profiles",
        "Empirical_Evidence": "SwinUNETR Protocol A attained 1.12% calibrated ECE while exhibiting 0.0% LGG recall; ResNet18 Protocol C attained 0.99% calibrated ECE with 41.67% recall.",
        "Evidence_Strength": "DEMONSTRATED",
        "Potential_Overclaim": "Declaring CNNs superior to Transformers for medical imaging.",
        "Recommended_Wording": "State that reliability is an emergent property from the interaction of data, architecture, loss, and thresholds, and that high calibration quality does not guarantee minority sensitivity.",
    },
    {
        "Claim_Topic": "4.7 Methodological vs Clinical recommendations",
        "Empirical_Evidence": "Operating points demonstrate continuous trade-offs (e.g. tau=0.35 yielding 66.7% recall with 4 false alarms).",
        "Evidence_Strength": "METHODOLOGICAL",
        "Potential_Overclaim": "Recommending a specific threshold for clinical practice or claiming the model is clinically ready.",
        "Recommended_Wording": "Frame as methodological recommendations for clinical AI evaluation pipelines, and emphasize that operational thresholds require prospective validation under pre-specified utility functions.",
    },
]

df_claim_audit = pd.DataFrame(claim_audit_records)
df_claim_audit.to_markdown = lambda: None # Avoid missing tabulate

# Format markdown table
headers = ["Claim Topic", "Empirical Evidence", "Evidence Strength", "Potential Overclaim", "Recommended Wording"]
audit_md = "# Discussion Section Scientific Claim-Evidence Audit\n\n"
audit_md += "This audit evaluates the scientific strength, potential overclaims, and recommended conservative wording for all major assertions in the Discussion section.\n\n"
audit_md += "| " + " | ".join(headers) + " |\n"
audit_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
for r in claim_audit_records:
    vals = [r["Claim_Topic"], r["Empirical_Evidence"], r["Evidence_Strength"], r["Potential_Overclaim"], r["Recommended_Wording"]]
    audit_md += "| " + " | ".join(vals) + " |\n"

(NOTES_DIR / "Discussion_claim_audit.md").write_text(audit_md)

print("[DONE] Discussion section draft and Claim Audit created successfully.")

if __name__ == "__main__":
    pass
