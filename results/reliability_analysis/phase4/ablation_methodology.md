# Phase 4 — Ablation Study #1: Methodology & Controlled Experimental Design

## 1. Research Question
*"Does class-weighting strategy consistently improve minority-class classification, or is its effect architecture-dependent?"*

## 2. Experimental Design & Controlled Scope
This ablation study is conducted by comparative analysis of the three pre-existing canonical experimental protocols:
- **Independent Variable**: Class-Weighting Strategy:
  - **Protocol A**: Unweighted CrossEntropyLoss (`class_weights = None`).
  - **Protocol B**: Canonical inverse-frequency class weighting (`[1.9018, 0.0982]`).
  - **Protocol C**: Stronger minority weighting (`[1.9207, 0.0793]`, 1.25× emphasis).
- **Controlled Fixed Variables**:
  - Random seed (`42`)
  - Dataset split (Train=876, Val=187, Test=188; LGG=12, HGG=176)
  - Input spatial resolution ($96 \times 96 \times 96$)
  - Model architectures (ResNet18 MedicalNet, DenseNet121 3D, SwinUNETR 3D)
  - Preprocessing, DataLoaders, and evaluation pipeline.

---

## 3. Explicit Methodological Disclaimer
> *"The class-weighting ablation is performed by comparative analysis of the three pre-existing canonical protocols, whose principal experimental distinction is the class-weighting strategy."*
> No new training runs were launched; zero modifications were made to archived checkpoints or test manifests.
