# Step 13 — Phase 3 Ablation Preparation (Read-Only Planning)

> **STRICT SAFETY ENFORCEMENT**: NO TRAINING HAS BEEN EXECUTED. THIS IS READ-ONLY ABLATION PLANNING.

---

## 1. Candidate Ablation Question #1
*"Does removing MedicalNet 3D transfer learning initialization degrade minority-class reliability and calibration, even when class weighting is applied?"*

## 2. Proposed Experimental Setup for Future Ablation #1
- **Component to Test/Remove**: Pretrained MedicalNet 3D weights (compare ResNet18 MedicalNet vs ResNet18 initialized randomly from scratch).
- **What Remains Fixed**: Dataset split, input resolution (96³), optimizer, learning rate, loss function, evaluation code, locked test loader.
- **Expected Hypothesis**: Randomly initialized 3D ResNet18 will fail to learn robust LGG representations on small cohorts, collapsing LGG recall and worsening ECE, similar to SwinUNETR.
- **Required Evaluation**: Locked test inference + Phase 2 ECE & Brier evaluation.

---

## 3. Status
Execution paused awaiting explicit user authorization.
