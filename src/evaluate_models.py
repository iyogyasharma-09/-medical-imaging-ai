"""
evaluate_models.py
------------------
Evaluate single-model, simple-average, and reliability-weighted ensemble
on the BraTS 2021 held-out test split.

Metrics per configuration:
  Accuracy, F1, ROC-AUC, ECE (Expected Calibration Error)

Outputs saved to results/:
  evaluation_summary.json       — all metrics (machine-readable)
  confusion_<config>.png        — confusion matrices
  calibration_<config>.png      — reliability / calibration diagram
  comparison_table.txt          — ASCII table comparing all configurations

Usage:
    python evaluate.py
    python evaluate.py --label-col label_grade
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on any machine
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVICE, EVALUATION_PATH, GRADCAM_DIR, RESULTS_DIR

try:
    from .train_models import create_dataloaders
    from .uncertainty import ReliabilityWeightedEnsemble, compute_ece, mc_dropout_predict
except ImportError:
    from train_models import create_dataloaders
    from uncertainty import ReliabilityWeightedEnsemble, compute_ece, mc_dropout_predict


# ------------------------------------------------------------------ #
# Plotting helpers                                                    #
# ------------------------------------------------------------------ #

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
    class_names: List[str] = ("LGG", "HGG"),
) -> None:
    cm     = confusion_matrix(y_true, y_pred)
    n      = len(class_names)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True label", xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _plot_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies  = (predictions == labels).astype(float)

    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_acc    = []
    bin_conf   = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_acc.append(accuracies[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.2)
    ax.bar(bin_mids, bin_acc, width=1.0 / n_bins, alpha=0.6,
           color="steelblue", label="Accuracy per bin")
    ax.plot(bin_conf, bin_acc, "o-", color="darkorange", label="Mean confidence")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy",
           title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ------------------------------------------------------------------ #
# Metrics helper                                                      #
# ------------------------------------------------------------------ #

def _summarize(bundle: Dict) -> Dict:
    y_true = np.array(bundle["y_true"])
    y_prob = np.array(bundle["y_prob"])
    y_pred = y_prob.argmax(axis=1)
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":  float(roc_auc_score(y_true, y_prob[:, 1])) if has_both_classes else 0.0,
        "ece":      float(compute_ece(y_prob, y_true)),
        "_y_true":  y_true,
        "_y_pred":  y_pred,
        "_y_prob":  y_prob,
    }


def _plot_coverage_accuracy(
    y_true: np.ndarray,
    uncertainties: np.ndarray,
    save_path: Path,
    title: str = "Coverage-Accuracy Tradeoff",
) -> Dict:
    """
    Plot coverage-accuracy tradeoff: as we reject high-uncertainty predictions
    (lower coverage), accuracy on retained predictions improves.

    This directly demonstrates the value of uncertainty estimation:
    - x-axis: coverage fraction (1.0 = use all predictions, 0.0 = reject all)
    - y-axis: accuracy on retained (non-rejected) subset

    A monotonically increasing curve shows the model's uncertainty
    signal correctly identifies uncertain (likely wrong) samples.

    Returns dict of coverage/accuracy arrays for JSON serialization.
    """
    # Sort samples from most certain (lowest uncertainty) to least certain
    order = np.argsort(uncertainties)      # ascending — most certain first
    sorted_true        = y_true[order]
    sorted_predictions = y_true[order]     # placeholder; compute from labels
    # We use the sorted y_true and derive prediction correctness from the
    # ensemble probabilities — here we approximate using the retained samples.
    # In practice 'uncertainties' comes from the weighted ensemble per sample.
    n = len(y_true)
    coverages  = []
    accuracies = []

    # Evaluate at 20 evenly-spaced coverage thresholds
    for k in np.linspace(n, 1, 20, dtype=int):
        retained_indices = order[:k]
        retained_true    = y_true[retained_indices]
        # We report accuracy as NaN if only one class is in retained set
        if len(np.unique(retained_true)) < 2:
            acc = float(np.nan)
        else:
            # Accuracy proxy: fraction of correctly ordered samples
            # (exact value computed by caller who has both y_pred and y_true)
            acc = float(np.nan)
        coverages.append(float(k / n))
        accuracies.append(acc)

    # Note: caller should pass (y_true, uncertainties, y_pred) for exact accuracy.
    # This version plots the coverage axis only — see evaluate() for the full call.
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(coverages, accuracies, "o-", color="royalblue", linewidth=2)
    ax.axhline(float(np.nanmean(accuracies)), color="gray", linestyle="--",
               label="Mean accuracy", linewidth=1)
    ax.set(xlabel="Coverage (fraction of test set retained)",
           ylabel="Accuracy on retained set",
           title=title, xlim=(0, 1.05), ylim=(0, 1.05))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return {"coverages": coverages, "accuracies": accuracies}


def _plot_coverage_accuracy_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    save_path: Path,
    title: str = "Coverage-Accuracy Tradeoff (Reliability-Weighted Ensemble)",
) -> Dict:
    """
    Full coverage-accuracy curve with exact accuracy per coverage level.
    Requires y_pred (predicted class) alongside y_true and uncertainties.
    """
    order  = np.argsort(uncertainties)   # most certain first
    n      = len(y_true)
    steps  = np.unique(np.linspace(1, n, 30, dtype=int))
    coverages, accuracies = [], []

    for k in steps:
        retained = order[:k]
        acc      = float(accuracy_score(y_true[retained], y_pred[retained]))
        coverages.append(float(k / n))
        accuracies.append(acc)

    full_acc = float(accuracy_score(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, accuracies, "o-", color="royalblue", linewidth=2,
            label="Accuracy (most-certain first)")
    ax.axhline(full_acc, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Full test accuracy = {full_acc:.3f}")
    ax.fill_between(coverages, full_acc, accuracies,
                    where=[a > full_acc for a in accuracies],
                    alpha=0.15, color="green", label="Gain from uncertainty filtering")
    ax.set(xlabel="Coverage (fraction of test set retained)",
           ylabel="Accuracy on retained set",
           title=title, xlim=(0, 1.05), ylim=(0, 1.05))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    result = {
        "coverages":         coverages,
        "accuracies":        accuracies,
        "full_test_accuracy": full_acc,
        "max_gain":          float(max(a - full_acc for a in accuracies)),
    }
    (save_path.parent / "coverage_accuracy.json").write_text(json.dumps(result, indent=2))
    return result



# ------------------------------------------------------------------ #
# Main evaluation                                                     #
# ------------------------------------------------------------------ #

def evaluate(label_col: str = "label_grade") -> Dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[evaluate] Loading test split (label_col='{label_col}') …")
    _, _, test_loader, _ = create_dataloaders(label_col=label_col)

    ensemble = ReliabilityWeightedEnsemble()
    if not ensemble.models:
        raise RuntimeError(
            "No trained 3D model checkpoints found.\n"
            "Run training on a GPU machine first: python train.py"
        )

    # Accumulators
    model_outputs    = {name: {"y_true": [], "y_prob": []} for name in ensemble.models}
    simple_ensemble  = {"y_true": [], "y_prob": []}
    weighted_ensmbl  = {"y_true": [], "y_prob": []}

    print(f"[evaluate] Running inference on {len(test_loader)} test batches …")
    for batch in tqdm(test_loader):
        inputs    = batch["image"].to(DEVICE)
        labels_np = batch["label"].numpy()
        batch_probs: List[np.ndarray] = []

        for model_name, model in ensemble.models.items():
            pred  = mc_dropout_predict(model, inputs)
            probs = pred.mean_probabilities
            model_outputs[model_name]["y_true"].extend(labels_np.tolist())
            model_outputs[model_name]["y_prob"].extend(probs.tolist())
            batch_probs.append(probs)

        # Simple average
        simple_probs = np.stack(batch_probs, axis=0).mean(axis=0)
        simple_ensemble["y_true"].extend(labels_np.tolist())
        simple_ensemble["y_prob"].extend(simple_probs.tolist())

        # Reliability-weighted
        weighted     = ensemble.predict_tensor(inputs)
        w_probs      = np.array(weighted["probabilities"], dtype=np.float32)
        weighted_ensmbl["y_true"].extend(labels_np.tolist())
        weighted_ensmbl["y_prob"].extend(w_probs.tolist())
        # Collect per-sample uncertainty = max normalized uncertainty across models
        # (proxy: mean confidence complemented — max(1 - max_prob) per sample)
        per_sample_uncertainty = (1.0 - w_probs.max(axis=1)).tolist()
        if "uncertainty" not in weighted_ensmbl:
            weighted_ensmbl["uncertainty"] = []
        weighted_ensmbl["uncertainty"].extend(per_sample_uncertainty)

    # Compute metrics
    all_bundles: Dict[str, Dict] = {}
    for name, bundle in model_outputs.items():
        all_bundles[name] = _summarize(bundle)
    all_bundles["simple_average_ensemble"]      = _summarize(simple_ensemble)
    all_bundles["reliability_weighted_ensemble"] = _summarize(weighted_ensmbl)

    # ---- Figures ----
    for config_name, stats in all_bundles.items():
        safe_name = config_name.replace(" ", "_")

        _plot_confusion_matrix(
            stats["_y_true"], stats["_y_pred"],
            title=f"Confusion — {config_name}",
            save_path=RESULTS_DIR / f"confusion_{safe_name}.png",
        )
        _plot_calibration(
            stats["_y_prob"], stats["_y_true"],
            title=f"Calibration — {config_name}",
            save_path=RESULTS_DIR / f"calibration_{safe_name}.png",
        )

    # ---- ASCII comparison table ----
    header = f"\n{'Configuration':<42} {'Acc':>6} {'F1':>6} {'AUC':>6} {'ECE':>6}"
    sep    = "-" * len(header)
    rows   = [header, sep]
    for name, stats in all_bundles.items():
        rows.append(
            f"{name:<42} {stats['accuracy']:>6.3f} {stats['f1']:>6.3f} "
            f"{stats['roc_auc']:>6.3f} {stats['ece']:>6.4f}"
        )
    table = "\n".join(rows)
    print(table)
    (RESULTS_DIR / "comparison_table.txt").write_text(table)

    # ---- JSON output (strip internal numpy arrays) ----
    def _clean(s: Dict) -> Dict:
        return {k: v for k, v in s.items() if not k.startswith("_")}

    summary = {
        "single_models":                {n: _clean(s) for n, s in all_bundles.items()
                                          if n not in ("simple_average_ensemble",
                                                        "reliability_weighted_ensemble")},
        "simple_average_ensemble":      _clean(all_bundles["simple_average_ensemble"]),
        "reliability_weighted_ensemble": _clean(all_bundles["reliability_weighted_ensemble"]),
        "evaluated_models":             list(ensemble.models.keys()),
        "label_col":                    label_col,
    }
    # ---- Coverage-accuracy tradeoff (publishable experiment) ----
    # Collect per-sample uncertainty from the weighted ensemble output
    w_y_true        = np.array(weighted_ensmbl["y_true"])
    w_y_prob        = np.array(weighted_ensmbl["y_prob"])
    w_y_pred        = w_y_prob.argmax(axis=1)
    w_uncertainties = np.array(weighted_ensmbl.get("uncertainty", []))
    if len(w_uncertainties) == len(w_y_true):
        coverage_result = _plot_coverage_accuracy_full(
            y_true=w_y_true,
            y_pred=w_y_pred,
            uncertainties=w_uncertainties,
            save_path=RESULTS_DIR / "coverage_accuracy.png",
        )
        summary["coverage_accuracy"] = coverage_result
        print(f"  Max accuracy gain from filtering: +{coverage_result['max_gain']:.3f}")
    else:
        print("  [coverage-accuracy] Uncertainty not collected — skipping curve.")
        print("  (Per-sample uncertainty is logged from predict_tensor in future runs.)")

    EVALUATION_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n[evaluate] Results saved to {RESULTS_DIR}/")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate 3D BraTS 2021 ensemble")
    parser.add_argument("--label-col", type=str, default="label_grade",
                        choices=["label_grade", "label_idh"])
    args = parser.parse_args()
    print(json.dumps(evaluate(label_col=args.label_col), indent=2))
