"""Ensemble-level hyperparameter tuning.

Strategy: two-stage tuning
--------------------------
Fitting all base models is the expensive part of an ensemble.  Re-doing it
for every Optuna trial is prohibitive.  Instead we use a two-stage approach:

  Stage 1 — fit each base model once on ``X_dict_train``.
  Stage 2 — collect their predictions on ``X_dict_val``, then run an Optuna
             study that tunes only the *combination* mechanism (weights,
             meta_alpha) against those fixed predictions.

This decouples the slow part (base model training) from the fast part
(meta-learner / weight search), making 50-100 trials practical.

After the study, the ensemble is fully re-fitted with the best params so it
is ready for inference.

What gets tuned per ensemble type
----------------------------------
VotingEnsemble:   per-model weights (sampled on a simplex via normalised floats)
BlendingEnsemble: meta_alpha of the Ridge / LogisticRegression meta-model
StackingEnsemble: meta_alpha (n_folds is a structural decision, not tuned here)

Usage
-----
::

    from molgate.models.factory import create_model
    from molgate.models.tuning import tune_ensemble

    ensemble = create_model("ensemble_voting_fp", task_type="regression")
    best_params = tune_ensemble(
        ensemble,
        X_dict_train, y_train,
        X_dict_val,   y_val,
        n_trials=50,
    )
    # ensemble is now fully fitted with best params
    preds = ensemble.predict(X_dict_test)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression, Ridge

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def tune_ensemble(
    ensemble: Any,
    X_dict_train: dict[str, Any],
    y_train: np.ndarray,
    X_dict_val: dict[str, Any],
    y_val: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
    metric: str | None = None,
) -> dict[str, Any]:
    """Tune ensemble combination hyperparameters using a two-stage strategy.

    Parameters
    ----------
    ensemble : VotingEnsemble | BlendingEnsemble | StackingEnsemble
        An unfitted (or previously fitted) ensemble instance created by
        ``create_model()`` or directly.  The ensemble's base models are
        fitted once internally; the combination mechanism is then searched
        with Optuna.
    X_dict_train : dict[str, Any]
        Training features for each base model (numpy arrays or graph lists).
    y_train : np.ndarray
        Training targets.
    X_dict_val : dict[str, Any]
        Validation features — used to score candidate combination params.
    y_val : np.ndarray
        Validation targets.
    n_trials : int
        Number of Optuna trials (default 50).
    seed : int
        Random seed for Optuna sampler and internal splits.
    metric : str, optional
        Scoring metric: "mae", "rmse", or "auroc".  Defaults to "mae" for
        regression and "auroc" for classification (inferred from ensemble).

    Returns
    -------
    dict[str, Any]
        Best hyperparameters found.  The ensemble is also re-fitted in place
        with these params and is ready for ``predict()``.

    Raises
    ------
    TypeError
        If ``ensemble`` is not a recognised ensemble class.
    """
    from molgate.models.ensemble import (
        BlendingEnsemble,
        StackingEnsemble,
        VotingEnsemble,
        _fit_model,
        _get_predictions,
    )

    task_type = getattr(ensemble, "task_type", "regression")
    if metric is None:
        metric = "mae" if task_type == "regression" else "auroc"

    logger.info(
        f"Tuning {type(ensemble).__name__} "
        f"({len(ensemble.models)} base models, metric={metric}, n_trials={n_trials})"
    )

    # -----------------------------------------------------------------------
    # Stage 1: fit each base model on the training set (once)
    # -----------------------------------------------------------------------
    logger.info("Stage 1: fitting base models on training data...")
    for name, model in ensemble.models.items():
        logger.info(f"  Fitting: {name}")
        _fit_model(model, X_dict_train[name], y_train)

    # -----------------------------------------------------------------------
    # Stage 2: collect val predictions from each base model (fixed)
    # -----------------------------------------------------------------------
    logger.info("Stage 2: collecting validation predictions...")
    model_names = list(ensemble.models.keys())
    val_preds: dict[str, np.ndarray] = {
        name: _get_predictions(model, X_dict_val[name], task_type)
        for name, model in ensemble.models.items()
    }
    val_pred_matrix = np.column_stack([val_preds[n] for n in model_names])
    logger.info(f"  Val meta-feature matrix: {val_pred_matrix.shape}")

    # -----------------------------------------------------------------------
    # Stage 3: Optuna study over combination params
    # -----------------------------------------------------------------------
    if isinstance(ensemble, VotingEnsemble):
        best_params = _tune_voting_weights(
            val_pred_matrix, y_val, model_names, task_type, metric, n_trials, seed
        )
    elif isinstance(ensemble, (BlendingEnsemble, StackingEnsemble)):
        best_params = _tune_meta_alpha(
            val_pred_matrix, y_val, task_type, metric, n_trials, seed
        )
    else:
        raise TypeError(
            f"Unsupported ensemble type: {type(ensemble).__name__}. "
            f"Expected VotingEnsemble, BlendingEnsemble, or StackingEnsemble."
        )

    logger.info(f"Best params found: {best_params}")

    # -----------------------------------------------------------------------
    # Stage 4: full re-fit with best params
    # -----------------------------------------------------------------------
    logger.info("Stage 4: full re-fit with best params...")
    _apply_best_params(ensemble, best_params)
    _full_refit(ensemble, X_dict_train, y_train, X_dict_val, y_val)

    return best_params


# ---------------------------------------------------------------------------
# Per-type tuning logic
# ---------------------------------------------------------------------------

def _tune_voting_weights(
    val_pred_matrix: np.ndarray,
    y_val: np.ndarray,
    model_names: list[str],
    task_type: str,
    metric: str,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Optimise per-model weights for VotingEnsemble.

    Samples weights on the (n_models - 1)-simplex by suggesting n_models
    floats in [0, 1] and normalising.  This keeps the search space bounded
    while allowing any weight distribution.
    """
    n_models = len(model_names)

    def objective(trial: optuna.Trial) -> float:
        raw = np.array([
            trial.suggest_float(f"w_{name}", 0.0, 1.0)
            for name in model_names
        ])
        total = raw.sum()
        weights = (raw / total) if total > 0 else np.ones(n_models) / n_models
        avg = val_pred_matrix @ weights
        return _score(avg, y_val, task_type, metric)

    study = optuna.create_study(
        direction="minimize" if metric in ("mae", "rmse") else "maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    raw_best = np.array([study.best_params[f"w_{n}"] for n in model_names])
    total = raw_best.sum()
    weights_best = (raw_best / total).tolist() if total > 0 else [1.0 / n_models] * n_models

    logger.info(
        f"  Best weights: {dict(zip(model_names, [round(w, 4) for w in weights_best]))} "
        f"| {metric}={study.best_value:.4f}"
    )
    return {"weights": weights_best, "model_names": model_names}


def _tune_meta_alpha(
    val_pred_matrix: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
    metric: str,
    n_trials: int,
    seed: int,
) -> dict[str, Any]:
    """Optimise meta_alpha for BlendingEnsemble or StackingEnsemble.

    For each trial, a fresh Ridge (regression) or LogisticRegression
    (classification) is fit on val_pred_matrix → y_val and scored.
    This is extremely fast since the base model predictions are fixed.
    """
    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("meta_alpha", 1e-3, 100.0, log=True)
        if task_type == "classification":
            meta = LogisticRegression(C=1.0 / alpha, max_iter=500, solver="lbfgs")
        else:
            meta = Ridge(alpha=alpha)
        meta.fit(val_pred_matrix, y_val)
        preds = meta.predict(val_pred_matrix)
        return _score(preds, y_val, task_type, metric)

    study = optuna.create_study(
        direction="minimize" if metric in ("mae", "rmse") else "maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_alpha = study.best_params["meta_alpha"]
    logger.info(
        f"  Best meta_alpha={best_alpha:.4f} | {metric}={study.best_value:.4f}"
    )
    return {"meta_alpha": best_alpha}


# ---------------------------------------------------------------------------
# Apply best params + full re-fit
# ---------------------------------------------------------------------------

def _apply_best_params(ensemble: Any, best_params: dict) -> None:
    """Push best params back onto the ensemble object."""
    from molgate.models.ensemble import VotingEnsemble, BlendingEnsemble, StackingEnsemble

    if isinstance(ensemble, VotingEnsemble):
        ensemble.weights = best_params["weights"]

    elif isinstance(ensemble, (BlendingEnsemble, StackingEnsemble)):
        ensemble.meta_alpha = best_params["meta_alpha"]


def _full_refit(
    ensemble: Any,
    X_dict_train: dict[str, Any],
    y_train: np.ndarray,
    X_dict_val: dict[str, Any],
    y_val: np.ndarray,
) -> None:
    """Re-fit the ensemble end-to-end with updated params.

    VotingEnsemble / StackingEnsemble: fit on train only.
    BlendingEnsemble: needs both train and val.
    """
    from molgate.models.ensemble import BlendingEnsemble

    # Reset fitted state so the ensemble trains fresh base models
    ensemble._fitted = False
    for model in ensemble.models.values():
        if hasattr(model, "_fitted"):
            model._fitted = False

    if isinstance(ensemble, BlendingEnsemble):
        ensemble.fit(X_dict_train, y_train, X_dict_val, y_val)
    else:
        ensemble.fit(X_dict_train, y_train)


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def _score(y_pred: np.ndarray, y_true: np.ndarray, task_type: str, metric: str) -> float:
    """Compute a scalar score from predictions.  Lower = better for MAE/RMSE."""
    if metric == "mae":
        return float(np.mean(np.abs(y_pred - y_true)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    if metric == "auroc":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric!r}. Use 'mae', 'rmse', or 'auroc'.")


# ---------------------------------------------------------------------------
# Convenience: tune all ensemble types and compare
# ---------------------------------------------------------------------------

def tune_and_compare(
    ensembles: dict[str, Any],
    X_dict_train: dict[str, Any],
    y_train: np.ndarray,
    X_dict_val: dict[str, Any],
    y_val: np.ndarray,
    X_dict_test: dict[str, Any],
    y_test: np.ndarray,
    n_trials: int = 50,
    seed: int = 42,
    metric: str | None = None,
) -> "pd.DataFrame":
    """Tune multiple ensembles and return a comparison DataFrame.

    Parameters
    ----------
    ensembles : dict[str, ensemble]
        Named ensemble instances to tune and compare.
    X_dict_train / y_train : training data
    X_dict_val / y_val     : validation data (used for tuning)
    X_dict_test / y_test   : held-out test data (reported in final table)
    n_trials : int
        Optuna trials per ensemble (default 50).
    metric : str, optional
        Scoring metric for tuning.  Defaults to "mae" (regression).

    Returns
    -------
    pd.DataFrame
        Columns: name, best_params, val_{metric}, test_mae, test_rmse, test_r2.
    """
    import pandas as pd
    from molgate.training.metrics import compute_metrics

    rows = []
    for name, ensemble in ensembles.items():
        logger.info(f"\n{'='*50}\nTuning: {name}\n{'='*50}")
        task_type = getattr(ensemble, "task_type", "regression")
        eff_metric = metric or ("mae" if task_type == "regression" else "auroc")

        best_params = tune_ensemble(
            ensemble,
            X_dict_train, y_train,
            X_dict_val,   y_val,
            n_trials=n_trials,
            seed=seed,
            metric=eff_metric,
        )

        val_preds  = ensemble.predict(X_dict_val)
        test_preds = ensemble.predict(X_dict_test)
        val_score  = _score(val_preds,  y_val,  task_type, eff_metric)
        test_metrics = compute_metrics(y_test, test_preds, task_type=task_type)

        row = {
            "name":        name,
            "best_params": str(best_params),
            f"val_{eff_metric}": round(val_score, 4),
        }
        row.update({f"test_{k}": round(v, 4) for k, v in test_metrics.items()})
        rows.append(row)
        logger.info(f"  Test metrics: {test_metrics}")

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = f"test_{eff_metric}"
        if sort_col in df.columns:
            df = df.sort_values(sort_col).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.featurizer import compute_descriptors, compute_fingerprints
    from molgate.data.loaders import load_tdc_benchmark_split
    from molgate.models.factory import create_model

    print("Loading solubility TDC benchmark split...")
    train_val_df, test_df = load_tdc_benchmark_split("solubility")

    # Carve out a val set from train_val for tuning
    from molgate.data.splits import random_split
    sub = random_split(train_val_df, val_frac=0.1, test_frac=0.0, seed=42)
    train_df = sub["train"]
    val_df   = sub["val"]

    def _featurize(df):
        sm = df["smiles"].tolist()
        return {
            "lgbm_morgan":       compute_fingerprints(sm),
            "lgbm_descriptors":  compute_descriptors(sm).values,
            "rf_morgan":         compute_fingerprints(sm),
            "ridge_descriptors": compute_descriptors(sm).values,
        }

    X_train = _featurize(train_df)
    X_val   = _featurize(val_df)
    X_test  = _featurize(test_df)
    y_train = train_df["y"].values
    y_val   = val_df["y"].values
    y_test  = test_df["y"].values

    ensembles = {
        "voting_fp": create_model("ensemble_voting_fp",  task_type="regression"),
        "blending":  create_model("ensemble_blending",   task_type="regression"),
        "stacking":  create_model("ensemble_stacking",   task_type="regression"),
    }

    print("\nTuning all ensembles (n_trials=30)...")
    results_df = tune_and_compare(
        ensembles,
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        n_trials=30,
        metric="mae",
    )

    print(f"\n{'='*60}")
    print("Tuning Results")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))

    import IPython
    IPython.embed()
