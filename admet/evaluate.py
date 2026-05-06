"""Evaluation metrics and inference utilities for ADMET property models.

Regression: RMSE, MAE, R², Spearman correlation.
Classification: AUROC, AUPRC, balanced accuracy.

All metric functions accept numpy arrays (not tensors) to keep post-training
analysis free of PyG/CUDA dependencies.

predict_aligned() is the leaderboard-specific variant: it returns predictions
for every row in the original DataFrame, filling positions where featurization
failed with a fallback value (training-set mean for regression, 0.5 for
classification). This ensures prediction arrays match the length of TDC's
fixed test DataFrames exactly.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from admet.dataset import ADMETDataset
from admet.model import ADMETModel


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted values, shape (N,).

    Returns:
        Dict with keys: rmse, mae, r2, spearman.
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    spearman, _ = spearmanr(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": float(r2), "spearman": float(spearman)}


def evaluate_classification(
    y_true: np.ndarray, y_scores: np.ndarray
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Binary ground truth labels, shape (N,).
        y_scores: Predicted probabilities (after sigmoid), shape (N,).

    Returns:
        Dict with keys: auroc, auprc, balanced_accuracy.
    """
    auroc = float(roc_auc_score(y_true, y_scores))
    auprc = float(average_precision_score(y_true, y_scores))
    y_pred_binary = (y_scores >= 0.5).astype(int)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred_binary))
    return {"auroc": auroc, "auprc": auprc, "balanced_accuracy": bal_acc}


def predict(
    model: ADMETModel,
    dataset: ADMETDataset,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on successfully featurized molecules.

    For classification models, y_pred contains sigmoid probabilities.
    For regression models, y_pred contains raw scalar predictions.

    Args:
        model: Trained ADMETModel (must already be on the correct device).
        dataset: Dataset to run inference on.
        batch_size: DataLoader batch size.

    Returns:
        (y_true, y_pred) both shape (N_valid,) where N_valid ≤ len(dataset._df).
        Molecules that failed featurization are excluded.
    """
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).squeeze(-1)
            preds = torch.sigmoid(logits) if dataset.task_type == "classification" else logits
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.squeeze(-1).cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def predict_aligned(
    model: ADMETModel,
    dataset: ADMETDataset,
    fill_value: float | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Run inference and return predictions aligned with the original DataFrame.

    Returns one prediction per row in dataset._df (the original TDC DataFrame
    before featurization). Positions where featurization failed are filled with
    fill_value. This is required for leaderboard submission where every test
    molecule must have a prediction.

    Args:
        model: Trained ADMETModel.
        dataset: Dataset whose ._df and ._valid_positions attributes track
            which rows featurized successfully.
        fill_value: Value to insert at failed positions. Defaults to the mean
            of successfully predicted values (regression) or 0.5 (classification).
        batch_size: DataLoader batch size.

    Returns:
        Predictions array of shape (dataset._n_total,), aligned with dataset._df.
    """
    _, y_pred = predict(model, dataset, batch_size=batch_size)

    if fill_value is None:
        fill_value = 0.5 if dataset.task_type == "classification" else float(y_pred.mean())

    aligned = np.full(dataset._n_total, fill_value, dtype=np.float32)
    for list_idx, df_pos in enumerate(dataset._valid_positions):
        aligned[df_pos] = y_pred[list_idx]

    return aligned


def evaluate(
    model: ADMETModel,
    dataset: ADMETDataset,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run inference and compute all metrics for the dataset's task type.

    Convenience wrapper around predict() + evaluate_regression/classification().
    Only evaluates on molecules that featurized successfully.
    """
    y_true, y_pred = predict(model, dataset, batch_size=batch_size)
    if dataset.task_type == "regression":
        return evaluate_regression(y_true, y_pred)
    return evaluate_classification(y_true, y_pred)
