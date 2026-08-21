"""
run_all.py
----------
[LEGACY - BraTS 2021 Pipeline Entry Point]

NOTE: For active BraTS 2023 research, execute the pipeline steps directly:
  1. python preprocess_brats2023.py
  2. python train_brats2023.py
  3. python evaluate_models.py
  4. python advanced_analysis.py
  5. python plot_results.py

Steps (in order for BraTS 2021 legacy):
  1. Dataset audit     (always — no GPU required)
  2. Training          (skip with --skip-train if checkpoints already exist)
  3. Evaluation        (skip with --skip-eval)
  4. Grad-CAM export   (skip with --skip-gradcam)
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (BRATS_LABELS_CSV, BRATS_SUBJECTS_DIR, EPOCHS,
                    BATCH_SIZE, MODELS, create_directories)


def _banner(step: str, total: str) -> None:
    print("\n" + "╔" + "═" * 62 + "╗")
    print(f"║  STEP {step}/{total} — {total and ''}{step}{'':>40}║".replace(
        f" STEP {step}/{total} — {total and ''}{step}{'':>40}",
        f" STEP {step}  {total}"
    ))
    print("╚" + "═" * 62 + "╝")


def run_audit(csv: str, data_root: str) -> None:
    print("\n" + "━" * 65)
    print("  STEP 1 / 4 — Dataset Audit")
    print("━" * 65)
    from src.dataset_audit import run_audit
    report = run_audit(Path(csv), Path(data_root))
    verdict = report.get("verdict", {})
    if not verdict.get("grade_task_ok"):
        print("\n[run_all] ✗ Audit failed — grade task is not viable. Stopping.")
        sys.exit(1)
    if not verdict.get("splits_clean"):
        print("\n[run_all] ✗ Data leakage detected — fix splits before training. Stopping.")
        sys.exit(1)


def run_training(models, epochs, batch_size, label_col) -> None:
    print("\n" + "━" * 65)
    print("  STEP 2 / 4 — Training")
    print("━" * 65)
    from src.train_models import train_all_models
    t0 = time.time()
    summary = train_all_models(
        model_names=models, epochs=epochs,
        batch_size=batch_size, label_col=label_col,
    )
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min.")
    for name, res in summary.items():
        print(f"    {name:<30} best_val_acc={res['best_val_accuracy']:.4f}")


def run_evaluation(label_col) -> None:
    print("\n" + "━" * 65)
    print("  STEP 3 / 4 — Evaluation")
    print("━" * 65)
    from src.evaluate_models import evaluate
    summary = evaluate(label_col=label_col)
    # Print comparison table
    rwe = summary.get("reliability_weighted_ensemble", {})
    print(f"\n  Reliability-Weighted Ensemble →  "
          f"Acc={rwe.get('accuracy', 0):.3f}  "
          f"F1={rwe.get('f1', 0):.3f}  "
          f"AUC={rwe.get('roc_auc', 0):.3f}  "
          f"ECE={rwe.get('ece', 0):.4f}")


def run_gradcam(n_subjects, label_col) -> None:
    print("\n" + "━" * 65)
    print("  STEP 4 / 4 — Grad-CAM Export")
    print("━" * 65)
    from gradcam_export import run_gradcam_export
    run_gradcam_export(n_subjects=n_subjects, label_col=label_col)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BraTS 2021 — Full 3D glioma grading pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument("--csv",       default=str(BRATS_LABELS_CSV),    help="Path to labels CSV")
    parser.add_argument("--data_root", default=str(BRATS_SUBJECTS_DIR),  help="Subject folder root")
    parser.add_argument("--label-col", default="label_grade",
                        choices=["label_grade", "label_idh"])

    # Skip flags
    parser.add_argument("--skip-train",   action="store_true", help="Skip training (use existing checkpoints)")
    parser.add_argument("--skip-eval",    action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-gradcam", action="store_true", help="Skip Grad-CAM export")

    # Training config
    parser.add_argument("--models",      nargs="+", default=None, choices=list(MODELS.keys()))
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--batch-size",  type=int, default=BATCH_SIZE)

    # Grad-CAM config
    parser.add_argument("--n-subjects",  type=int, default=4,
                        help="Number of subjects for Grad-CAM (balanced per class)")

    args = parser.parse_args()
    create_directories()

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  BraTS 2021 — 3D Reliability-Aware Ensemble Pipeline        ║")
    print("╚" + "═" * 62 + "╝")
    print(f"  label_col  : {args.label_col}")
    print(f"  skip_train : {args.skip_train}")
    print(f"  skip_eval  : {args.skip_eval}")
    print(f"  skip_gradcam: {args.skip_gradcam}")

    # Step 1 — always run audit
    run_audit(args.csv, args.data_root)

    # Step 2 — training
    if not args.skip_train:
        run_training(args.models, args.epochs, args.batch_size, args.label_col)
    else:
        print("\n  [run_all] Training skipped (--skip-train). Using existing checkpoints.")

    # Step 3 — evaluation
    if not args.skip_eval:
        run_evaluation(args.label_col)
    else:
        print("\n  [run_all] Evaluation skipped (--skip-eval).")

    # Step 4 — Grad-CAM
    if not args.skip_gradcam:
        run_gradcam(args.n_subjects, args.label_col)
    else:
        print("\n  [run_all] Grad-CAM export skipped (--skip-gradcam).")

    print("\n" + "━" * 65)
    print("  Pipeline complete.")
    print("━" * 65 + "\n")


if __name__ == "__main__":
    main()
