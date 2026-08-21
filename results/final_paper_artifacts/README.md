# Final Publication Artifacts Directory

This directory contains the definitive, consolidated, publication-ready tables, figures, evidence matrices, and manuscript drafts for the BraTS 2023 GLI Trustworthy Evaluation study.

## Directory Structure & Provenance

- `01_master_results/`: Consolidated CSV and Markdown master tables containing all raw, calibrated, and threshold-swept results. (Source: `results/posthoc_calibration_comparison/`).
- `02_calibration/`: Post-hoc temperature scaling tables and metadata. (Source: `results/posthoc_calibration_comparison/temperature_parameters.json`).
- `03_threshold_analysis/`: Calibrated operating-point tables across thresholds [0.10, 0.50]. (Source: `results/posthoc_calibration_comparison/multi_model_threshold_comparison.csv`).
- `04_focal_analysis/`: Dedicated Focal Loss comparison table and analysis. (Source: `results/focal_locked_test/`).
- `05_figures/`: Publication-quality multi-model reliability diagrams and operating characteristic curves.
- `06_tables/`: Formatted markdown and CSV tables for Manuscript Tables 1, 2, 3, and 4.
- `07_claim_evidence/`: Comprehensive claim-evidence matrix classifying all paper claims.
- `08_manuscript_notes/`: Drafts for Results (Section 3) and Discussion Outline (Section 4).
- `09_experiment_inventory/`: Definitive catalogue of all 10 completed experimental stages.

## Integrity Verification
- **All results are derived from existing, frozen experimental outputs.**
- **Zero models were retrained; zero weights or baseline files were modified.**
