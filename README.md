# BraTS 2023 MRI Classification Pipeline

This repository contains a full 3D Deep Learning pipeline for predicting Glioma (GLI) grading (LGG vs. HGG) using the **BraTS 2023 Adult Glioma** dataset. 

It handles raw 3D NIfTI MRI data (`t1c`, `t1n`, `t2f`, `t2w` sequences), stacking them into multi-channel volumes, and training state-of-the-art 3D architectures using MONAI and PyTorch. 

## Key Features

* **3D Volumetric Processing:** Efficiently loads and transforms 4-channel 3D MRI scans (`96x96x96` resolution).
* **Models Evaluated:**
  * **ResNet18-MedicalNet:** Pretrained on large-scale medical 3D datasets.
  * **DenseNet121-3D:** Strong baseline with excellent feature reuse.
  * **Swin UNETR (3D Swin Transformer):** Advanced self-attention architecture adapted for 3D classification.
* **Mixed Precision (AMP) & Multi-processing:** Optimized data loading (`NUM_WORKERS=4` with Windows `spawn` fix) and memory-efficient training for RTX A1000 (8GB VRAM).
* **OpenVINO Inference:** Scripts for exporting PyTorch models to ONNX and OpenVINO IR for real-time edge inference on Intel Arc iGPUs and NPUs.
* **Advanced Evaluation:** Includes threshold tuning, AUC-weighted soft ensembling, calibration reliability diagrams (ECE/Brier), and 5-fold cross-validation.

## Results Summary (GLI task - LGG vs HGG)

* **DenseNet121-3D:** Achieved **95.2% accuracy** and **0.88 AUC** with an optimal decision threshold of 0.17.
* **Ensemble:** An AUC-weighted soft ensemble reached an **AUC of 0.919**.
* **NPU / iGPU benchmark:** Inference runs in real-time (~365ms/subject) on Intel Arc iGPU using OpenVINO.

## Repository Structure

* `train_brats2023.py`: Main training loop with registry updating and artifact saving.
* `evaluate_models.py`: Computes test-set metrics (Accuracy, AUC, F1, Sensitivity, Specificity).
* `advanced_analysis.py`: Performs CV, ensemble, calibration, and threshold tuning.
* `plot_results.py`: Generates comprehensive matplotlib dashboards and curves (`results/plots/`).
* `benchmark_all_models.py` & `export_and_infer_npu.py`: OpenVINO conversion and latency benchmarking.
* `results/`: Contains evaluation JSONs and generated plots.
* `models/`: Stores exported ONNX and OpenVINO checkpoints (PyTorch weights `.pt` are ignored by git).
* `dataset/`: Directory for BraTS 2023 NIfTI files (ignored by git to prevent massive uploads).

## Usage (Active BraTS 2023 Pipeline)

1. **Prepare Data:** Extract BraTS 2023 data into `dataset/` and run `python preprocess_brats2023.py` to generate manifests.
2. **Train:** `python train_brats2023.py`
3. **Evaluate:** `python evaluate_models.py`
4. **Analyze:** `python advanced_analysis.py`
5. **Plot:** `python plot_results.py`

*Note: Root scripts `run_all.py`, `train.py`, `evaluate.py`, and `preprocess_all.py` are legacy BraTS 2021 entry points preserved for backward compatibility and should not be used for BraTS 2023 experiments.*
