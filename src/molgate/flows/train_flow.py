"""Prefect flow: train a single model on a single dataset variant.

This is the workhorse flow — called directly for one-off training runs
and mapped over a matrix of conditions by ``bias_experiment.py``.

Pipeline:
    1. Load dataset
    2. Optionally apply bias (scaffold, property range, target region, etc.)
    3. Run bias diagnostics (if biased)
    4. Split into train/val/test
    5. Featurize (fingerprints, descriptors, or graphs depending on model)
    6. Train the model
    7. Evaluate on the test set
    8. Log everything to W&B

Design decisions:
    - **Model-agnostic featurization**: The flow reads ``feature_type`` from
      the model's metadata (set by the factory) and calls the appropriate
      featurizer.  FingerprintModel gets numpy arrays, GNN gets PyG graphs.
    - **Bias as a config dict**: Bias is specified as ``{"method": "scaffold",
      "params": {"top_n": 10}}``.  If ``None`` or ``{"method": "unbiased"}``,
      no bias is applied.
    - **Deterministic**: All randomness goes through ``seed``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from prefect import flow, task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bias application
# ---------------------------------------------------------------------------

# Mapping from bias method names to (function, param_names).
# Lazy imports inside the functions to avoid circular imports.
_BIAS_METHODS = {
    "scaffold": "bias_by_scaffold",
    "property_range": "bias_by_property_range",
    "target_region": "bias_by_target_region",
    "substructure": "bias_by_substructure",
    "cluster": "bias_by_cluster",
}


def _apply_bias(df: pd.DataFrame, bias_config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply a bias function to a DataFrame based on config.

    Parameters
    ----------
    df : pd.DataFrame
        Full clean dataset.
    bias_config : dict
        ``{"method": "scaffold", "params": {"top_n": 10}}``.

    Returns
    -------
    biased_df : pd.DataFrame
        The biased subset.
    bias_metadata : dict
        Metadata from the bias function (what was removed, stats, etc.).
    """
    import molgate.data.bias as bias_module

    method = bias_config["method"]
    params = bias_config.get("params", {})

    if method not in _BIAS_METHODS:
        raise ValueError(
            f"Unknown bias method: {method!r}. "
            f"Available: {sorted(_BIAS_METHODS.keys())}"
        )

    fn_name = _BIAS_METHODS[method]
    fn = getattr(bias_module, fn_name)
    biased_df, metadata = fn(df, **params)
    return biased_df, metadata


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load_dataset", retries=2, retry_delay_seconds=10)
def task_load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load and clean a TDC ADMET dataset."""
    from molgate.data.loaders import load_dataset
    df = load_dataset(dataset_name)
    logger.info(f"Loaded {dataset_name}: {len(df)} molecules")
    return df


@task(name="apply_bias")
def task_apply_bias(
    df: pd.DataFrame,
    bias_config: dict | None,
) -> tuple[pd.DataFrame, dict | None]:
    """Apply bias to the dataset (or pass through if unbiased).

    Returns (working_df, bias_metadata).  If no bias is applied,
    bias_metadata is None.
    """
    if bias_config is None or bias_config.get("method") == "unbiased":
        logger.info("No bias applied (unbiased)")
        return df, None

    method = bias_config["method"]
    params = bias_config.get("params", {})
    logger.info(f"Applying bias: {method} with params={params}")

    biased_df, metadata = _apply_bias(df, bias_config)
    logger.info(
        f"  Bias result: {len(biased_df)}/{len(df)} molecules retained "
        f"({len(biased_df)/len(df):.1%})"
    )
    return biased_df, metadata


@task(name="bias_diagnostics")
def task_bias_diagnostics(
    original_df: pd.DataFrame,
    biased_df: pd.DataFrame,
    bias_name: str,
) -> dict:
    """Run bias diagnostics comparing original to biased dataset."""
    from molgate.analysis.bias_diagnostics import bias_report

    report = bias_report(
        smiles_original=original_df["smiles"].tolist(),
        smiles_biased=biased_df["smiles"].tolist(),
        target_original=original_df["y"].values,
        target_biased=biased_df["y"].values,
        bias_name=bias_name,
        run_adversarial=True,
    )
    logger.info(
        f"  Bias diagnostics: retention={report.get('retention_rate', 0):.1%}, "
        f"adversarial AUROC={report.get('adversarial', {}).get('auroc_mean', 'N/A')}"
    )
    return report


@task(name="split_dataset")
def task_split_dataset(
    df: pd.DataFrame,
    split_type: str = "random",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split dataset into train/val/test."""
    from molgate.data.splits import random_split, scaffold_split

    if split_type == "scaffold":
        splits = scaffold_split(df, seed=seed)
    else:
        splits = random_split(df, seed=seed)

    for name, split_df in splits.items():
        logger.info(f"  {name}: {len(split_df)} molecules")
    return splits


@task(name="featurize")
def task_featurize(
    splits: dict[str, pd.DataFrame],
    feature_type: str,
    fp_radius: int = 2,
    fp_nbits: int = 2048,
) -> dict[str, Any]:
    """Featurize all splits according to the model's feature type.

    Returns a dict with keys like:
        - "train_X", "val_X", "test_X" (numpy arrays for FP/descriptor models)
        - "train_graphs", "val_graphs", "test_graphs" (PyG Data lists for GNN)
        - "train_y", "val_y", "test_y" (target arrays)
        - "train_smiles", "val_smiles", "test_smiles" (SMILES lists)
        - "feature_type" (for downstream reference)
    """
    from molgate.data.featurizer import (
        compute_descriptors,
        compute_fingerprints,
        smiles_list_to_graphs,
    )

    result: dict[str, Any] = {"feature_type": feature_type}

    for split_name in ("train", "val", "test"):
        split_df = splits[split_name]
        smiles = split_df["smiles"].tolist()
        y = split_df["y"].values

        result[f"{split_name}_y"] = y
        result[f"{split_name}_smiles"] = smiles

        if feature_type == "morgan":
            result[f"{split_name}_X"] = compute_fingerprints(
                smiles, radius=fp_radius, n_bits=fp_nbits,
            )
        elif feature_type == "descriptors":
            result[f"{split_name}_X"] = compute_descriptors(smiles).values
        elif feature_type == "graph":
            result[f"{split_name}_graphs"] = smiles_list_to_graphs(
                smiles, y.tolist(),
            )
        else:
            raise ValueError(f"Unknown feature_type: {feature_type!r}")

    logger.info(f"Featurized all splits (type={feature_type})")
    return result


@task(name="create_model")
def task_create_model(
    model_name: str,
    task_type: str,
    overrides: dict | None = None,
):
    """Create a model from the factory."""
    from molgate.models.factory import create_model
    model = create_model(model_name, task_type=task_type, overrides=overrides)
    logger.info(f"Created model: {model_name} (task={task_type})")
    return model


@task(name="train_model")
def task_train_model(
    model,
    features: dict[str, Any],
    task_type: str,
    wandb_run=None,
) -> tuple[Any, dict]:
    """Train the model and return (trained_model_or_trainer, train_info).

    For FingerprintModel: calls model.fit(X, y).
    For GNN: creates a Trainer from model.training_config and runs fit().

    Returns a tuple of (model_or_trainer, info_dict) where the first
    element has a .predict() method ready for evaluation.
    """
    feature_type = features["feature_type"]

    if feature_type in ("morgan", "descriptors"):
        # FingerprintModel path
        X_train = features["train_X"]
        y_train = features["train_y"]
        model.fit(X_train, y_train)
        logger.info(f"FingerprintModel trained on {X_train.shape[0]} samples")
        return model, {"model_type": "fingerprint", "n_train": X_train.shape[0]}

    elif feature_type == "graph":
        # GNN path — needs a Trainer
        from molgate.training.trainer import Trainer

        training_config = getattr(model, "training_config", {})
        trainer = Trainer.from_config(model, training_config, task_type=task_type)

        # Attach W&B run if provided
        if wandb_run is not None:
            trainer.wandb_run = wandb_run

        train_graphs = features["train_graphs"]
        val_graphs = features["val_graphs"]
        history = trainer.fit(train_graphs, val_graphs)

        logger.info(
            f"GNN trained: {len(history.train_loss)} epochs, "
            f"best val loss={history.best_val_loss:.4f} at epoch {history.best_epoch}"
        )
        return trainer, {
            "model_type": "gnn",
            "n_train": len(train_graphs),
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
            "training_time_seconds": history.training_time_seconds,
            "total_epochs": len(history.train_loss),
        }
    else:
        raise ValueError(f"Unknown feature_type: {feature_type!r}")


@task(name="evaluate_model")
def task_evaluate_model(
    trained_model,
    features: dict[str, Any],
    task_type: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the trained model on the test set."""
    from molgate.training.evaluate import evaluate_model

    feature_type = features["feature_type"]

    if feature_type in ("morgan", "descriptors"):
        test_data = features["test_X"]
    else:
        test_data = features["test_graphs"]

    metrics, pred_df = evaluate_model(
        model=trained_model,
        test_data=test_data,
        y_true=features["test_y"],
        task_type=task_type,
        smiles=features["test_smiles"],
    )
    logger.info(
        f"Test evaluation: "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    return metrics, pred_df


@task(name="log_to_wandb")
def task_log_to_wandb(
    dataset_name: str,
    model_name: str,
    task_type: str,
    split_type: str,
    seed: int,
    bias_config: dict | None,
    bias_metadata: dict | None,
    bias_diagnostics: dict | None,
    train_info: dict,
    test_metrics: dict[str, float],
    predictions_df: pd.DataFrame,
    wandb_mode: str = "disabled",
) -> None:
    """Log the complete training run to W&B."""
    from molgate.tracking import build_tags, init_run, log_metrics, log_predictions_table

    bias_name = bias_config["method"] if bias_config else "unbiased"
    bias_params_str = str(bias_config.get("params", {})) if bias_config else ""

    run = init_run(
        name=f"{model_name}_{dataset_name}_{bias_name}",
        config={
            "dataset": dataset_name,
            "model": model_name,
            "task_type": task_type,
            "split_type": split_type,
            "seed": seed,
            "bias_method": bias_name,
            "bias_params": bias_params_str,
            **train_info,
        },
        tags=build_tags(
            dataset=dataset_name,
            model=model_name,
            bias=bias_name,
            split=split_type,
        ),
        group=f"train_{dataset_name}",
        job_type="train",
        mode=wandb_mode,
    )

    try:
        # Test metrics as summary
        run.summary.update({f"test_{k}": v for k, v in test_metrics.items()})

        # Bias diagnostics
        if bias_diagnostics:
            for k, v in bias_diagnostics.items():
                if isinstance(v, (int, float)):
                    run.summary[f"bias_{k}"] = v

        # Predictions table
        log_predictions_table(run, predictions_df)

        logger.info(f"Logged training run to W&B: {run.name}")

    finally:
        run.finish()


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="train_flow", log_prints=True)
def train_flow(
    dataset_name: str = "solubility",
    model_name: str = "lgbm_morgan",
    task_type: str = "regression",
    bias_config: dict | None = None,
    split_type: str = "random",
    seed: int = 42,
    model_overrides: dict | None = None,
    wandb_mode: str = "disabled",
) -> dict[str, Any]:
    """Train a single model on a single dataset variant.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (must be in the registry).
    model_name : str
        Model name from models.yaml (e.g., "lgbm_morgan", "gnn").
    task_type : str
        "regression" or "classification".
    bias_config : dict, optional
        Bias specification: ``{"method": "scaffold", "params": {"top_n": 10}}``.
        If None, the full (unbiased) dataset is used.
    split_type : str
        "random" or "scaffold".
    seed : int
        Random seed.
    model_overrides : dict, optional
        Override model hyperparameters (passed to factory).
    wandb_mode : str
        W&B mode: "online", "offline", or "disabled".

    Returns
    -------
    dict
        Run summary with keys: metrics, predictions_df, train_info,
        bias_metadata, bias_diagnostics.
    """
    bias_name = bias_config["method"] if bias_config else "unbiased"
    logger.info(
        f"Train flow: {model_name} on {dataset_name} "
        f"(bias={bias_name}, split={split_type}, seed={seed})"
    )

    # 1. Load dataset
    full_df = task_load_dataset(dataset_name)

    # 2. Apply bias (if any)
    working_df, bias_metadata = task_apply_bias(full_df, bias_config)

    # 3. Bias diagnostics (only if bias was applied)
    bias_diagnostics_result = None
    if bias_metadata is not None:
        bias_diagnostics_result = task_bias_diagnostics(
            original_df=full_df,
            biased_df=working_df,
            bias_name=bias_name,
        )

    # 4. Split
    splits = task_split_dataset(working_df, split_type=split_type, seed=seed)

    # 5. Create model (to read its feature_type metadata)
    model = task_create_model(model_name, task_type, overrides=model_overrides)

    # Determine feature type from model metadata
    if hasattr(model, "feature_type"):
        if model.feature_type == "descriptors":
            feature_type = "descriptors"
        else:
            feature_type = "morgan"
        fp_radius = getattr(model, "fp_radius", 2)
        fp_nbits = getattr(model, "fp_nbits", 2048)
    else:
        # GNN
        feature_type = "graph"
        fp_radius, fp_nbits = 2, 2048

    # 6. Featurize
    features = task_featurize(
        splits, feature_type=feature_type,
        fp_radius=fp_radius, fp_nbits=fp_nbits,
    )

    # 7. Train
    trained_model, train_info = task_train_model(
        model, features, task_type,
    )

    # 8. Evaluate
    test_metrics, predictions_df = task_evaluate_model(
        trained_model, features, task_type,
    )

    # 9. Log to W&B
    task_log_to_wandb(
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        split_type=split_type,
        seed=seed,
        bias_config=bias_config,
        bias_metadata=bias_metadata,
        bias_diagnostics=bias_diagnostics_result,
        train_info=train_info,
        test_metrics=test_metrics,
        predictions_df=predictions_df,
        wandb_mode=wandb_mode,
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "bias": bias_name,
        "split": split_type,
        "seed": seed,
        "metrics": test_metrics,
        "predictions_df": predictions_df,
        "train_info": train_info,
        "bias_metadata": bias_metadata,
        "bias_diagnostics": bias_diagnostics_result,
    }


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # === Run 1: Unbiased LightGBM + Morgan FPs ===
    print("=" * 60)
    print("Run 1: LightGBM + Morgan FPs (unbiased)")
    print("=" * 60)

    result_fp = train_flow(
        dataset_name="solubility",
        model_name="lgbm_morgan",
        task_type="regression",
        bias_config=None,
        split_type="random",
        seed=42,
        wandb_mode="disabled",
    )

    print(f"\nMetrics: {result_fp['metrics']}")

    # === Run 2: Biased (scaffold top 10) ===
    print("\n" + "=" * 60)
    print("Run 2: LightGBM + Morgan FPs (scaffold bias, top 10)")
    print("=" * 60)

    result_biased = train_flow(
        dataset_name="solubility",
        model_name="lgbm_morgan",
        task_type="regression",
        bias_config={"method": "scaffold", "params": {"top_n": 10}},
        split_type="random",
        seed=42,
        wandb_mode="disabled",
    )

    print(f"\nMetrics: {result_biased['metrics']}")

    # === Run 3: GNN (unbiased, quick) ===
    print("\n" + "=" * 60)
    print("Run 3: GNN (unbiased, 20 epochs)")
    print("=" * 60)

    result_gnn = train_flow(
        dataset_name="solubility",
        model_name="gnn",
        task_type="regression",
        bias_config=None,
        split_type="random",
        seed=42,
        model_overrides={"training": {"epochs": 20, "patience": 10}},
        wandb_mode="disabled",
    )

    print(f"\nMetrics: {result_gnn['metrics']}")

    # === Comparison ===
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  {'Model':<25s}  {'RMSE':>8s}  {'MAE':>8s}  {'R²':>8s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
    for label, r in [
        ("LGBM unbiased", result_fp),
        ("LGBM scaffold_top10", result_biased),
        ("GNN unbiased (20ep)", result_gnn),
    ]:
        m = r["metrics"]
        print(f"  {label:<25s}  {m['rmse']:>8.4f}  {m['mae']:>8.4f}  {m['r2']:>8.4f}")

    import IPython
    IPython.embed()
