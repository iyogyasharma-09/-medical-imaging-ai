# Methods Section Scientific Claim & Implementation Audit

This audit verifies that every methodological detail in the Methods draft corresponds strictly to actual code and verified artifacts in the repository.

| Section | Method Claim | Evidence Artifact | Implementation Status | Leakage Risk |
| --- | --- | --- | --- | --- |
| 2.1 Study Design | 5-pillar evaluation separating ranking, stratified sensitivity, calibration, thresholds, uncertainty | evaluate_locked_test.py, run_multi_model_temperature_scaling.py | VERIFIED (Codebase & Artifacts) | None (Framework design) |
| 2.2 Dataset & Splits | BraTS 2023 GLI: Train=876, Val=187 (10 LGG), Test=188 (12 LGG) | val_manifest.json, test_manifest.json, config.py | VERIFIED (Exact Match) | None (Disjoint manifest split) |
| 2.3 Input Representation | 4-channel 3D volume [T1, T1ce, T2, FLAIR] resized to 4x96x96x96 | transforms.py, config.py | VERIFIED (Exact Match) | None (Deterministic format) |
| 2.4 Preprocessing | RAS reorientation, 1mm spacing, non-zero foreground crop, z-score norm | imported_files/transforms.py (get_val_transforms) | VERIFIED (Exact Match) | None (Per-volume normalization) |
| 2.5 Architectures | ResNet18 MedicalNet, DenseNet121 3D, SwinUNETR 3D with Dropout=0.4 | train_brats2023.py (BackboneWithDropoutHead, SwinClassifier3D) | VERIFIED (Exact Match) | None (Standard constructors) |
| 2.6 Training Setup | AdamW (lr=1e-4, wd=1e-4), batch=2, AMP fp16, early stopping on Val Macro-F1 | train_brats2023.py, config.py | VERIFIED (Exact Match) | None (Val-only early stopping) |
| 2.7 Loss Protocols | Protocol A (Unweighted), B (Inv-Freq), C (Minority x1.25, normalized sum=K) | train_brats2023.py (compute_protocol_weights) | VERIFIED (Exact Match) | None (Train-only weight fit) |
| 2.8 Focal Loss | Focal Loss alpha=1.0, gamma=2.0 on ResNet18 MedicalNet | train_brats2023_focal.py (FocalLoss class) | VERIFIED (Exact Match) | None (Isolated experiment) |
| 2.9 Locked Test | Locked test evaluated with torch.no_grad() and model.eval() | evaluate_locked_test.py | VERIFIED (Exact Match) | None (Read-only evaluation) |
| 2.11 Temperature Scaling | T* optimized strictly on Validation NLL via bounded minimize_scalar, frozen before test | scratch/run_focal_temperature_scaling.py, temperature_parameters.json | VERIFIED (Exact Match) | None (Zero test-label exposure) |
| 2.13 Uncertainty | Predictive entropy H(P) = -sum(p * log2(p)) in bits | evaluate_locked_test.py (compute_uncertainty) | VERIFIED (Exact Match) | None (Post-hoc forward metric) |
