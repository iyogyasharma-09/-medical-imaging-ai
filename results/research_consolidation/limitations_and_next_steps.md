# Limitations & Next-Experiment Recommendation

## 1. Study Limitations
1. **Single Test Split**: All results were evaluated on a single 188-subject test split (12 LGG / 176 HGG).
2. **Minority Subgroup Size**: 12 LGG test cases reflect the natural BraTS 2023 imbalanced distribution, but yield wider variance on recall estimates ($\pm 1$ subject $= \pm 8.33\%$).
3. **No External Dataset**: Validation was performed exclusively on MICCAI BraTS 2023 cohorts.
4. **Comparative Ablation Scope**: Phase 4 analyzed pre-existing canonical protocols rather than newly retrained ablation variants.

---

## 2. Recommended Next Experiment (For Future Authorization)
- **Proposed Experiment**: **5-Fold Cross-Validation on Development Set** for ResNet18 MedicalNet (Protocol A) to quantify training fold variance.
- **Strict Protection**: Will use ONLY Train (876) + Val (187) cohorts; the 188-subject locked test set will remain completely untouched.
- **Compute Cost**: 5 training runs (~2 hours on GPU).
- **Status**: **PAUSED**. Execution will not begin without explicit human authorization.
