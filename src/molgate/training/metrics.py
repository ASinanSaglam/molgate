"""Metric functions for model evaluation.

Provides ``compute_metrics`` — a single entry point that computes all
relevant metrics for a given task type (regression or classification).

Design choice: pure functions, no state. Each metric is a standalone
function that takes (y_true, y_pred) and returns a float. The
``compute_metrics`` wrapper calls the appropriate set and returns a dict.

All functions handle edge cases (constant predictions, single-class
targets) gracefully by returning NaN rather than crashing.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt( mean( (y_true - y_pred)^2 ) )

    Same units as the target variable. For solubility (log mol/L),
    RMSE of 1.0 means predictions are off by ~1 log unit on average.
    This is the primary metric for our regression tasks.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    MAE = mean( |y_true - y_pred| )

    Less sensitive to outliers than RMSE (no squaring). If RMSE >> MAE,
    it means a few predictions have very large errors (outliers dominate
    RMSE but not MAE).
    """
    return float(mean_absolute_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²).

    R² = 1 - SS_res / SS_tot
       = 1 - Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²

    Interpretation:
        R² = 1.0  → perfect predictions
        R² = 0.0  → predictions are no better than predicting the mean
        R² < 0.0  → predictions are worse than the mean (possible with
                     bad models or mismatched train/test distributions)

    Returns NaN if all true values are identical (SS_tot = 0).
    """
    y_true = np.asarray(y_true)
    if np.std(y_true) == 0:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Measures linear correlation between predictions and true values.
    Unlike R², Pearson r is invariant to scale and shift — a model that
    predicts y_true * 2 + 5 gets Pearson r = 1.0 but R² < 1.0.

    Returns NaN if either array is constant.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Area Under the Receiver Operating Characteristic curve.

    y_pred should be probabilities (0 to 1), not hard labels.
    This is the primary metric for our classification tasks.

    Returns NaN if only one class is present in y_true (ROC curve
    is undefined without both positives and negatives).
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        logger.warning("AUROC undefined: only one class in y_true")
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_pred))
    except ValueError:
        return float("nan")


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Classification accuracy at a given probability threshold.

    Converts probabilities to hard labels using the threshold,
    then computes fraction correct.

    Note: accuracy is misleading for imbalanced datasets. A dataset
    with 95% negatives gets 95% accuracy from always predicting 0.
    Use AUROC as the primary metric instead.
    """
    y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred_binary))


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """F1 score (harmonic mean of precision and recall).

    F1 = 2 * (precision * recall) / (precision + recall)

    Balances false positives and false negatives. More informative
    than accuracy for imbalanced datasets.

    Returns 0.0 if either precision or recall is 0.
    """
    y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress zero_division warnings
        return float(f1_score(y_true, y_pred_binary, zero_division=0.0))


def binary_precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Precision: of all predicted positives, how many are truly positive?

    precision = TP / (TP + FP)

    High precision means few false alarms. Important when the cost of
    a false positive is high (e.g., advancing a toxic compound to trials).
    """
    y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(precision_score(y_true, y_pred_binary, zero_division=0.0))


def binary_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Recall (sensitivity): of all true positives, how many did we catch?

    recall = TP / (TP + FN)

    High recall means few missed positives. Important when the cost of
    missing a positive is high (e.g., failing to flag a toxic compound).
    """
    y_pred_binary = (np.asarray(y_pred) >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(recall_score(y_true, y_pred_binary, zero_division=0.0))


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    """Compute all relevant metrics for a task type.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions. For classification, these should be
        probabilities (not hard labels).
    task_type : str
        "regression" or "classification".

    Returns
    -------
    dict[str, float]
        Metric name → value. All values are floats (or NaN for
        undefined metrics).

    Examples
    --------
    >>> compute_metrics([1.0, 2.0, 3.0], [1.1, 2.2, 2.8], "regression")
    {'rmse': 0.173, 'mae': 0.166, 'r2': 0.959, 'pearson_r': 0.989}

    >>> compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], "classification")
    {'auroc': 1.0, 'accuracy': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")

    if task_type == "regression":
        metrics = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r_squared(y_true, y_pred),
            "pearson_r": pearson_r(y_true, y_pred),
        }
    elif task_type == "classification":
        metrics = {
            "auroc": auroc(y_true, y_pred),
            "accuracy": binary_accuracy(y_true, y_pred),
            "f1": binary_f1(y_true, y_pred),
            "precision": binary_precision(y_true, y_pred),
            "recall": binary_recall(y_true, y_pred),
        }
    else:
        raise ValueError(
            f"Unknown task type: {task_type!r}. "
            f"Use 'regression' or 'classification'."
        )

    logger.info(
        f"Metrics ({task_type}): "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    return metrics


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== Regression metrics (synthetic) ===")
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_good = np.array([1.1, 2.2, 2.8, 4.3, 4.9])
    y_pred_bad = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Predicts mean

    print("\nGood predictions:")
    metrics_good = compute_metrics(y_true_reg, y_pred_good, "regression")
    for k, v in metrics_good.items():
        print(f"  {k:12s} = {v:.4f}")

    print("\nBad predictions (always predict mean):")
    metrics_bad = compute_metrics(y_true_reg, y_pred_bad, "regression")
    for k, v in metrics_bad.items():
        print(f"  {k:12s} = {v:.4f}")

    print("\n=== Classification metrics (synthetic) ===")
    y_true_cls = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred_proba = np.array([0.1, 0.3, 0.9, 0.8, 0.7, 0.2, 0.6, 0.4])

    print("\nReasonable predictions:")
    metrics_cls = compute_metrics(y_true_cls, y_pred_proba, "classification")
    for k, v in metrics_cls.items():
        print(f"  {k:12s} = {v:.4f}")

    print("\n=== Edge cases ===")
    # Constant predictions
    print("\nConstant predictions (all 3.0):")
    m = compute_metrics([1, 2, 3], [3.0, 3.0, 3.0], "regression")
    print(f"  R² = {m['r2']:.4f} (should be negative — worse than mean)")
    print(f"  Pearson r = {m['pearson_r']} (should be NaN — constant pred)")

    # Single class
    print("\nSingle class in y_true:")
    m = compute_metrics([1, 1, 1], [0.5, 0.7, 0.9], "classification")
    print(f"  AUROC = {m['auroc']} (should be NaN — only one class)")

    import IPython
    IPython.embed()
