# Core Paper Finding #1: Class-Weighting Strategy is Architecture-Dependent

## 1. Primary Research Statement
> **"Class weighting does not universally improve minority-class classification; its effect is architecture-dependent within the evaluated 3D glioma grading configurations."**

---

## 2. Empirical Support

The locked test evaluation results demonstrate opposing effects across architecture types:

1. **Pretrained CNN Architectures (ResNet18 MedicalNet & DenseNet121 3D)**:
   - For **ResNet18 MedicalNet**, transitioning from unweighted Protocol A to strong minority weighting Protocol C resulted in a **-0.1958 drop in Macro-F1** ($0.8842 \rightarrow 0.6884$) and a **-0.3333 drop in LGG Recall** ($0.7500 \rightarrow 0.4167$).
   - For **DenseNet121 3D**, transitioning from Protocol A to Protocol C resulted in a **-0.1407 drop in Macro-F1** ($0.7934 \rightarrow 0.6527$) and a **-0.3334 drop in LGG Recall** ($0.6667 \rightarrow 0.3333$).

2. **Un-pretrained Vision Transformer (SwinUNETR 3D)**:
   - For **SwinUNETR 3D**, transitioning from Protocol A to Protocol C resulted in a **+0.1901 gain in Macro-F1** ($0.4835 \rightarrow 0.6736$) and a **+0.2500 gain in LGG Recall** ($0.0000 \rightarrow 0.2500$).

---

## 3. Explicit Refutation of Universal Claims
The general assumption that *"Adding class weights always improves minority classification"* is **UNSUPPORTED AND REFUTED** by this study. When rich spatial features are already established via pretraining (MedicalNet), loss re-weighting distorts optimal probability boundaries and induces false positives.
