"""Model evaluation — internal metrics and TDC benchmark scoring.

Two evaluation modes:

1. **Internal evaluation** (``evaluate_model``):
   Computes all metrics from ``metrics.py`` and builds a per-molecule
   predictions DataFrame for error analysis.  Used by bias experiments,
   model comparisons, and notebooks.  Works with any model type.

2. **TDC benchmark evaluation** (``evaluate_tdc``):
   Runs predictions on the official TDC test set and scores them with
   TDC's ``group.evaluate()``.  Returns the official metric (e.g., MAE
   for solubility).  Used to compare our models against TDC leaderboard
   entries.

The split exists because our bias experiments use custom splits (that's
the whole point), but we separately need to know how we score on TDC's
fixed evaluation protocol.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from molgate.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    test_data,
    y_true: np.ndarray,
    task_type: str,
    smiles: list[str] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a model and return aggregate metrics + per-molecule predictions.

    Works with both FingerprintModel (numpy arrays) and GNN (PyG graphs
    via Trainer.predict).  The caller is responsible for passing the
    correct data format and running inference — for GNNs, pass a
    ``Trainer`` instance as ``model`` (it has ``.predict(graphs)``).

    Parameters
    ----------
    model
        Any object with a ``.predict(test_data) -> np.ndarray`` method.
        For FingerprintModel: pass the model directly, test_data is a
        numpy feature matrix.
        For GNN: pass the ``Trainer`` instance, test_data is a list of
        PyG Data objects.
    test_data
        Input data for prediction.  np.ndarray for fingerprint models,
        list[Data] for GNNs.
    y_true : np.ndarray
        Ground truth target values, shape (n_samples,).
    task_type : str
        "regression" or "classification".
    smiles : list[str], optional
        SMILES strings corresponding to each sample.  If provided,
        included in the predictions DataFrame for error analysis.
        For GNN graphs, extracted from ``graph.smiles`` if not provided.

    Returns
    -------
    metrics : dict[str, float]
        Aggregate metrics (RMSE, MAE, R², Pearson r for regression;
        AUROC, accuracy, F1, precision, recall for classification).
    predictions_df : pd.DataFrame
        Per-molecule predictions with columns:
        - ``y_true``: ground truth
        - ``y_pred``: model prediction
        - ``error``: signed error (regression) or absolute error (classification)
        - ``smiles``: SMILES string (if available)
    """
    y_true = np.asarray(y_true, dtype=float)

    # Run inference
    y_pred = model.predict(test_data)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples, "
            f"y_pred has {len(y_pred)}"
        )

    # Compute aggregate metrics
    metrics = compute_metrics(y_true, y_pred, task_type)

    # Build per-molecule predictions DataFrame
    pred_dict: dict[str, Any] = {
        "y_true": y_true,
        "y_pred": y_pred,
    }

    # Error computation depends on task type
    if task_type == "regression":
        # Signed error: positive = over-prediction
        pred_dict["error"] = y_pred - y_true
        pred_dict["abs_error"] = np.abs(y_pred - y_true)
    else:
        # For classification: absolute difference between predicted prob and true label
        pred_dict["error"] = np.abs(y_pred - y_true)

    # Try to get SMILES from graphs if not provided
    if smiles is None and isinstance(test_data, list) and hasattr(test_data[0], "smiles"):
        smiles = [g.smiles for g in test_data]

    if smiles is not None:
        pred_dict["smiles"] = smiles

    predictions_df = pd.DataFrame(pred_dict)

    logger.info(
        f"Evaluation ({task_type}): {len(y_true)} samples, "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )

    return metrics, predictions_df


# ---------------------------------------------------------------------------
# TDC benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_tdc(
    model,
    benchmark_name: str,
    featurize_fn,
    group=None,
    data_path: str = "data/",
) -> dict[str, Any]:
    """Evaluate a model against the official TDC benchmark test set.

    Loads the TDC benchmark, featurizes the test set, runs inference,
    and scores predictions using TDC's official ``group.evaluate()``.

    Parameters
    ----------
    model
        Any object with a ``.predict(data) -> np.ndarray`` method.
        For GNNs, pass the ``Trainer`` instance.
    benchmark_name : str
        TDC benchmark name (e.g., "solubility_aqsoldb").
    featurize_fn : callable
        Function that converts ``(smiles_list, y_list) -> test_data``
        suitable for ``model.predict()``.  Examples:
        - For fingerprints: ``lambda s, y: compute_fingerprints(s)``
        - For GNN: ``lambda s, y: smiles_list_to_graphs(s, y)``
    group : BenchmarkGroup, optional
        Pre-initialized TDC BenchmarkGroup.  If None, creates one
        for the ADMET group.
    data_path : str
        Path where TDC stores downloaded data.

    Returns
    -------
    dict
        TDC evaluation result, e.g. ``{"solubility_aqsoldb": {"mae": 0.95}}``.
        Also includes our own metrics under the ``"molgate_metrics"`` key.
    """
    from tdc import BenchmarkGroup

    if group is None:
        group = BenchmarkGroup(name="ADMET_Group", path=data_path)

    benchmark = group.get(benchmark_name)
    test_df = benchmark["test"]

    # TDC test sets use "Drug" and "Y" column names
    test_smiles = test_df["Drug"].tolist()
    test_y = test_df["Y"].values

    # Featurize test set using the caller-provided function
    test_data = featurize_fn(test_smiles, test_y.tolist())

    # Run predictions
    y_pred = model.predict(test_data)
    y_pred = np.asarray(y_pred, dtype=float)

    # Official TDC evaluation
    predictions = {benchmark_name: y_pred}
    tdc_result = group.evaluate(predictions)

    # Also compute our own metrics for comparison
    our_metrics = compute_metrics(test_y, y_pred, "regression")

    logger.info(
        f"TDC evaluation ({benchmark_name}): "
        f"official={tdc_result[benchmark_name]}, "
        f"our_mae={our_metrics['mae']:.4f}, our_rmse={our_metrics['rmse']:.4f}"
    )

    return {
        "tdc_official": tdc_result[benchmark_name],
        "molgate_metrics": our_metrics,
        "benchmark_name": benchmark_name,
        "n_test_samples": len(test_y),
    }


def evaluate_tdc_multi_seed(
    train_fn,
    benchmark_name: str,
    featurize_fn,
    seeds: list[int] | None = None,
    data_path: str = "data/",
) -> dict[str, Any]:
    """Run TDC evaluation across multiple seeds (leaderboard protocol).

    TDC requires at least 5 independent runs to calculate mean and
    standard deviation for leaderboard submission.

    Parameters
    ----------
    train_fn : callable
        Function with signature ``train_fn(train_df, val_df, seed) -> model``
        that trains a model and returns it ready for prediction.
        ``train_df`` and ``val_df`` have columns: Drug, Y.
    benchmark_name : str
        TDC benchmark name (e.g., "solubility_aqsoldb").
    featurize_fn : callable
        Function that converts ``(smiles_list, y_list) -> test_data``.
    seeds : list[int], optional
        Random seeds for train/val splits.  Defaults to [1, 2, 3, 4, 5]
        per TDC convention.
    data_path : str
        Path where TDC stores downloaded data.

    Returns
    -------
    dict
        ``{"mean": float, "std": float, "per_seed": list[dict]}``.
        The mean/std are from TDC's ``evaluate_many``.
    """
    from tdc import BenchmarkGroup

    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    group = BenchmarkGroup(name="ADMET_Group", path=data_path)
    predictions_list = []

    for seed in seeds:
        benchmark = group.get(benchmark_name)
        name = benchmark["name"]
        test_df = benchmark["test"]

        train_df, val_df = group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed,
        )

        # Train model with this seed's split
        logger.info(f"Seed {seed}: training on {len(train_df)} samples, "
                     f"validating on {len(val_df)} samples")
        model = train_fn(train_df, val_df, seed)

        # Featurize and predict on test set
        test_smiles = test_df["Drug"].tolist()
        test_y = test_df["Y"].values
        test_data = featurize_fn(test_smiles, test_y.tolist())

        y_pred = model.predict(test_data)
        predictions_list.append({name: np.asarray(y_pred, dtype=float)})

    # TDC multi-seed evaluation returns {benchmark_name: [mean, std]}
    results = group.evaluate_many(predictions_list)
    mean_val, std_val = results[benchmark_name]

    logger.info(
        f"TDC multi-seed evaluation ({benchmark_name}): "
        f"mean={mean_val:.4f}, std={std_val:.4f} ({len(seeds)} seeds)"
    )

    return {
        "benchmark_name": benchmark_name,
        "mean": mean_val,
        "std": std_val,
        "seeds": seeds,
        "tdc_result": results,
    }


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.featurizer import compute_fingerprints, smiles_list_to_graphs
    from molgate.data.loaders import load_dataset
    from molgate.data.splits import random_split
    from molgate.models.baseline import FingerprintModel
    from molgate.models.gnn import MoleculeGNN
    from molgate.training.trainer import Trainer

    # --- Part 1: Internal evaluation with FingerprintModel ---
    print("=" * 60)
    print("Part 1: Internal evaluation — LightGBM + Morgan FPs")
    print("=" * 60)

    df = load_dataset("solubility")
    splits = random_split(df, seed=42)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    X_train = compute_fingerprints(train_df["smiles"].tolist())
    X_test = compute_fingerprints(test_df["smiles"].tolist())

    fp_model = FingerprintModel(
        task_type="regression",
        params={"n_estimators": 500, "learning_rate": 0.05, "verbose": -1},
    )
    fp_model.fit(X_train, train_df["y"].values)

    metrics, pred_df = evaluate_model(
        model=fp_model,
        test_data=X_test,
        y_true=test_df["y"].values,
        task_type="regression",
        smiles=test_df["smiles"].tolist(),
    )

    print(f"\n  Metrics: {metrics}")
    print(f"\n  Predictions (top 5):\n{pred_df.head()}")
    print(f"\n  Worst predictions:")
    print(pred_df.nlargest(5, "abs_error").to_string(index=False))

    # --- Part 2: Internal evaluation with GNN ---
    print("\n" + "=" * 60)
    print("Part 2: Internal evaluation — GNN")
    print("=" * 60)

    train_graphs = smiles_list_to_graphs(
        train_df["smiles"].tolist(), train_df["y"].tolist()
    )
    val_graphs = smiles_list_to_graphs(
        val_df["smiles"].tolist(), val_df["y"].tolist()
    )
    test_graphs = smiles_list_to_graphs(
        test_df["smiles"].tolist(), test_df["y"].tolist()
    )

    gnn_model = MoleculeGNN(hidden_dim=64, num_layers=2, task_type="regression")
    trainer = Trainer(
        gnn_model, task_type="regression",
        lr=0.001, epochs=30, patience=10, batch_size=32,
    )
    trainer.fit(train_graphs, val_graphs)

    metrics_gnn, pred_df_gnn = evaluate_model(
        model=trainer,
        test_data=test_graphs,
        y_true=test_df["y"].values,
        task_type="regression",
    )

    print(f"\n  Metrics: {metrics_gnn}")
    print(f"\n  Predictions (top 5):\n{pred_df_gnn.head()}")

    # --- Part 3: TDC benchmark evaluation ---
    print("\n" + "=" * 60)
    print("Part 3: TDC official evaluation — LightGBM")
    print("=" * 60)

    tdc_result = evaluate_tdc(
        model=fp_model,
        benchmark_name="solubility_aqsoldb",
        featurize_fn=lambda s, y: compute_fingerprints(s),
    )
    print(f"\n  TDC official: {tdc_result['tdc_official']}")
    print(f"\n  Our metrics:  {tdc_result['molgate_metrics']}")

    # --- Part 4: TDC benchmark evaluation — GNN ---
    print("\n" + "=" * 60)
    print("Part 4: TDC official evaluation — GNN")
    print("=" * 60)

    tdc_result_gnn = evaluate_tdc(
        model=trainer,
        benchmark_name="solubility_aqsoldb",
        featurize_fn=lambda s, y: smiles_list_to_graphs(s, y),
    )
    print(f"\n  TDC official: {tdc_result_gnn['tdc_official']}")
    print(f"\n  Our metrics:  {tdc_result_gnn['molgate_metrics']}")

    import IPython
    IPython.embed()
