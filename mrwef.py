"""
mrwef.py
--------
Multi-Model Reliability Weighted Evaluation Framework (MRWEF).

This standalone module implements Phase 1 of MRWEF: Reliability Score computation.

Reliability Score (R) combines three fundamental performance and stability metrics:
  1. F1 Score (performance metric: higher is better, range [0, 1])
  2. Expected Calibration Error (ECE) (calibration error: lower is better, range [0, 1])
  3. Coefficient of Variation (CV) (uncertainty/stability metric: lower is better, range [0, inf))

Formula:
  R = alpha * F1 + beta * (1.0 - ECE) + gamma * (1.0 / (1.0 + CV))

where alpha, beta, and gamma are configurable non-negative weights.
"""

from typing import Dict, Union


def validate_reliability_inputs(
    f1: float,
    ece: float,
    cv: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> None:
    """Validate inputs for Reliability Score computation.

    Raises:
        ValueError: If any metric or weight is outside valid ranges.
    """
    if not isinstance(f1, (int, float)) or not (0.0 <= float(f1) <= 1.0):
        raise ValueError(f"F1 score must be a float in range [0.0, 1.0], got {f1}")

    if not isinstance(ece, (int, float)) or not (0.0 <= float(ece) <= 1.0):
        raise ValueError(f"ECE must be a float in range [0.0, 1.0], got {ece}")

    if not isinstance(cv, (int, float)) or float(cv) < 0.0:
        raise ValueError(f"CV (Coefficient of Variation) must be non-negative, got {cv}")

    if not all(isinstance(w, (int, float)) and float(w) >= 0.0 for w in (alpha, beta, gamma)):
        raise ValueError(
            f"Weights alpha, beta, gamma must be non-negative numbers, got alpha={alpha}, beta={beta}, gamma={gamma}"
        )

    if float(alpha) == 0.0 and float(beta) == 0.0 and float(gamma) == 0.0:
        raise ValueError("At least one weight (alpha, beta, gamma) must be positive (> 0.0).")


def compute_reliability_score(
    f1: float,
    ece: float,
    cv: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    normalize_weights: bool = False,
) -> float:
    """Compute the MRWEF Reliability Score for a model.

    The Reliability Score integrates performance (F1), calibration (ECE),
    and prediction stability (CV) into a unified reliability metric.

    Formula:
      R = alpha * F1 + beta * (1.0 - ECE) + gamma * (1.0 / (1.0 + CV))

    Parameters:
        f1 (float): F1 Score in range [0.0, 1.0]. Higher is better.
        ece (float): Expected Calibration Error in range [0.0, 1.0]. Lower is better.
        cv (float): Coefficient of Variation (std / mean of predictions), >= 0.0. Lower is better.
        alpha (float): Weight for F1 component (default: 1.0).
        beta (float): Weight for Calibration component (default: 1.0).
        gamma (float): Weight for Stability component (default: 1.0).
        normalize_weights (bool): If True, normalizes (alpha, beta, gamma) to sum to 1.0.

    Returns:
        float: Computed Reliability Score.
    """
    validate_reliability_inputs(f1, ece, cv, alpha, beta, gamma)

    w_alpha = float(alpha)
    w_beta = float(beta)
    w_gamma = float(gamma)

    if normalize_weights:
        total_weight = w_alpha + w_beta + w_gamma
        w_alpha /= total_weight
        w_beta /= total_weight
        w_gamma /= total_weight

    f1_component = float(f1)
    calibration_component = 1.0 - float(ece)
    stability_component = 1.0 / (1.0 + float(cv))

    reliability_score = (
        w_alpha * f1_component +
        w_beta * calibration_component +
        w_gamma * stability_component
    )

    return float(reliability_score)


if __name__ == "__main__":
    # Smoke test / Demonstration
    sample_f1 = 0.952
    sample_ece = 0.034
    sample_cv = 0.120

    raw_score = compute_reliability_score(sample_f1, sample_ece, sample_cv)
    norm_score = compute_reliability_score(sample_f1, sample_ece, sample_cv, normalize_weights=True)

    print("=== MRWEF Phase 1: Reliability Score Computation ===")
    print(f"Input Metrics : F1={sample_f1}, ECE={sample_ece}, CV={sample_cv}")
    print(f"Default Score (alpha=1, beta=1, gamma=1)    : {raw_score:.4f}")
    print(f"Normalized Score (weights normalized to 1) : {norm_score:.4f}")
