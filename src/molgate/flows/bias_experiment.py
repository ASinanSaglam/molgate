"""Prefect flow: run the full bias experiment matrix.

This is the main event — the flow that answers the project's central
question: **"How does dataset composition affect molecular property
prediction models?"**

It reads ``configs/bias_experiments.yaml`` to build a matrix of:
    N datasets x M bias conditions x K model types

Then maps ``train_flow`` over every cell in the matrix, collecting
results into a single DataFrame for downstream comparison.

Design decisions:
    - **Sequential execution**: Each train_flow run is submitted
      sequentially rather than with Prefect's ``.submit()`` parallelism.
      Molecular featurization and GNN training are CPU/GPU-bound, so
      parallel runs would just thrash.  For real parallel execution,
      deploy with a Prefect work pool.
    - **Config-driven**: The experiment matrix is fully defined in YAML.
      Adding a new bias condition or model requires zero code changes.
    - **MW column handling**: The ``mw_narrow`` bias needs a MW column
      that doesn't exist in the raw data.  The flow computes descriptors
      and merges MW into the DataFrame before applying that specific bias.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from prefect import flow, task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "bias_experiments.yaml"
)


def load_experiment_config(config_path: str | Path | None = None) -> dict:
    """Load the bias experiment config from YAML.

    Returns
    -------
    dict
        Keys: ``bias_conditions``, ``experiment`` (datasets, models, seeds).
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    with open(path) as f:
        return yaml.safe_load(f)


def _build_bias_config(condition_name: str, condition: dict) -> dict | None:
    """Convert a YAML bias condition entry to a train_flow bias_config.

    Maps the YAML format (``bias_fn``, ``params``) to the train_flow
    format (``method``, ``params``).

    Returns None for unbiased conditions.
    """
    bias_fn = condition.get("bias_fn")
    if bias_fn is None:
        return None

    # Map bias function names to the short method names used by train_flow
    fn_to_method = {
        "bias_by_scaffold": "scaffold",
        "bias_by_property_range": "property_range",
        "bias_by_target_region": "target_region",
        "bias_by_substructure": "substructure",
        "bias_by_cluster": "cluster",
    }

    method = fn_to_method.get(bias_fn, bias_fn)
    params = dict(condition.get("params", {}))

    # Fix target_col casing: YAML has "Y" but our loaders produce "y"
    if "target_col" in params and params["target_col"] == "Y":
        params["target_col"] = "y"

    return {"method": method, "params": params}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="prepare_dataset")
def task_prepare_dataset(dataset_name: str) -> pd.DataFrame:
    """Load dataset and add computed descriptor columns (e.g., MW).

    Some bias conditions (like ``mw_narrow``) filter on molecular
    properties that aren't in the raw TDC data.  We compute RDKit
    descriptors and merge them so they're available for any bias
    function that needs them.
    """
    from molgate.data.featurizer import compute_descriptors
    from molgate.data.loaders import load_dataset

    df = load_dataset(dataset_name)

    # Compute descriptors and merge MW (and others) into the main df.
    # We use lowercase column names to match our descriptor naming.
    desc_df = compute_descriptors(df["smiles"].tolist())

    # Only merge columns that don't already exist
    for col in desc_df.columns:
        col_upper = col.upper()
        # Check both cases — bias config might reference "MW" or "mw"
        if col not in df.columns and col_upper not in df.columns:
            df[col] = desc_df[col].values

    logger.info(f"Prepared {dataset_name}: {len(df)} molecules, {len(df.columns)} columns")
    return df


@task(name="build_experiment_matrix")
def task_build_matrix(
    config: dict,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    conditions: list[str] | None = None,
) -> list[dict]:
    """Build the experiment matrix from config + optional overrides.

    Returns a list of run specifications, each a dict with keys:
    dataset, model, condition_name, bias_config, split_type, seed.
    """
    exp = config.get("experiment", {})
    all_conditions = config.get("bias_conditions", {})

    ds_list = datasets or exp.get("datasets", ["solubility"])
    model_list = models or exp.get("models", ["lgbm_morgan"])
    seed_list = seeds or exp.get("seeds", [42])
    cond_list = conditions or list(all_conditions.keys())

    matrix = []
    for dataset in ds_list:
        for cond_name in cond_list:
            if cond_name not in all_conditions:
                logger.warning(f"Unknown condition {cond_name!r}, skipping")
                continue
            condition = all_conditions[cond_name]
            bias_config = _build_bias_config(cond_name, condition)
            split_type = condition.get("split", "random")

            for model in model_list:
                for seed in seed_list:
                    matrix.append({
                        "dataset": dataset,
                        "model": model,
                        "condition_name": cond_name,
                        "bias_config": bias_config,
                        "split_type": split_type,
                        "seed": seed,
                    })

    logger.info(
        f"Experiment matrix: {len(ds_list)} datasets x {len(cond_list)} conditions "
        f"x {len(model_list)} models x {len(seed_list)} seeds = {len(matrix)} runs"
    )
    return matrix


@task(name="run_single_experiment")
def task_run_single(
    run_spec: dict,
    task_type: str,
    model_overrides: dict | None,
    wandb_mode: str,
) -> dict[str, Any]:
    """Execute a single train_flow run from a matrix specification.

    Wraps train_flow as a regular function call (not a sub-flow) to
    keep the Prefect task graph flat and readable.
    """
    from molgate.flows.train_flow import train_flow

    cond = run_spec["condition_name"]
    logger.info(
        f"Running: {run_spec['model']} / {run_spec['dataset']} / {cond} / seed={run_spec['seed']}"
    )

    result = train_flow(
        dataset_name=run_spec["dataset"],
        model_name=run_spec["model"],
        task_type=task_type,
        bias_config=run_spec["bias_config"],
        split_type=run_spec["split_type"],
        seed=run_spec["seed"],
        model_overrides=model_overrides,
        wandb_mode=wandb_mode,
    )

    # Attach condition name for aggregation
    result["condition"] = cond
    return result


@task(name="aggregate_results")
def task_aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate all run results into a single comparison DataFrame.

    Each row is one experiment run.  Columns include the experiment
    coordinates (dataset, model, condition, seed) and all metrics.
    """
    rows = []
    for r in results:
        row = {
            "dataset": r["dataset"],
            "model": r["model"],
            "condition": r["condition"],
            "bias": r["bias"],
            "split": r["split"],
            "seed": r["seed"],
        }
        # Flatten metrics
        for k, v in r.get("metrics", {}).items():
            row[k] = v
        # Training info
        for k, v in r.get("train_info", {}).items():
            if isinstance(v, (int, float, str)):
                row[f"train_{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Aggregated {len(df)} experiment results")
    return df


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="bias_experiment", log_prints=True)
def bias_experiment_flow(
    config_path: str | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    task_type: str = "regression",
    model_overrides: dict | None = None,
    wandb_mode: str = "disabled",
) -> pd.DataFrame:
    """Run the full bias experiment matrix.

    Parameters
    ----------
    config_path : str, optional
        Path to bias_experiments.yaml.  Defaults to ``configs/bias_experiments.yaml``.
    datasets : list[str], optional
        Override dataset list from config.
    models : list[str], optional
        Override model list from config.
    conditions : list[str], optional
        Override which bias conditions to run.  If None, runs all.
    seeds : list[int], optional
        Override seed list from config.
    task_type : str
        "regression" or "classification".
    model_overrides : dict, optional
        Override model hyperparameters (e.g., reduce GNN epochs for testing).
    wandb_mode : str
        W&B mode: "online", "offline", or "disabled".

    Returns
    -------
    pd.DataFrame
        Results table with one row per experiment run, containing
        experiment coordinates and all metrics.
    """
    # 1. Load experiment config
    config = load_experiment_config(config_path)
    logger.info("Loaded experiment config")

    # 2. Build the experiment matrix
    matrix = task_build_matrix(
        config,
        datasets=datasets,
        models=models,
        seeds=seeds,
        conditions=conditions,
    )

    # 3. Run each experiment sequentially
    results = []
    for i, run_spec in enumerate(matrix):
        cond = run_spec["condition_name"]
        logger.info(
            f"\n{'='*60}\n"
            f"Experiment {i+1}/{len(matrix)}: "
            f"{run_spec['model']} / {run_spec['dataset']} / {cond}\n"
            f"{'='*60}"
        )
        result = task_run_single(
            run_spec=run_spec,
            task_type=task_type,
            model_overrides=model_overrides,
            wandb_mode=wandb_mode,
        )
        results.append(result)

    # 4. Aggregate into a comparison DataFrame
    results_df = task_aggregate_results(results)

    logger.info(f"\nBias experiment complete: {len(results_df)} runs")
    return results_df


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Bias Experiment — Mini run (2 conditions x 2 models)")
    print("=" * 60)

    # Run a small subset for demo: 2 conditions x 2 models = 4 runs
    results_df = bias_experiment_flow(
        datasets=["solubility"],
        models=["lgbm_morgan", "lgbm_descriptors"],
        conditions=["unbiased", "scaffold_top10"],
        seeds=[42],
        task_type="regression",
        wandb_mode="disabled",
    )

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(results_df[["model", "condition", "rmse", "mae", "r2"]].to_string(index=False))

    # Pivot: rows=conditions, columns=models, values=RMSE
    if len(results_df) > 0:
        print(f"\n{'='*60}")
        print("RMSE Pivot (conditions x models)")
        print(f"{'='*60}")
        pivot = results_df.pivot_table(
            index="condition", columns="model", values="rmse",
        )
        print(pivot.to_string())

    import IPython
    IPython.embed()
