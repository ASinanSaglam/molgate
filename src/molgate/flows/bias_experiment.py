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
from prefect.task_runners import ConcurrentTaskRunner

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

def _enrich_with_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Merge computed descriptor columns (e.g., mw) into a DataFrame in-place.

    Only adds columns that don't already exist (checked case-insensitively).
    Used to make MW and other physicochemical properties available for
    bias conditions that filter on them.
    """
    from molgate.data.featurizer import compute_descriptors

    desc_df = compute_descriptors(df["smiles"].tolist())
    for col in desc_df.columns:
        if col not in df.columns and col.upper() not in df.columns:
            df[col] = desc_df[col].values
    return df


@task(name="prepare_dataset")
def task_prepare_dataset(
    dataset_name: str,
    use_tdc_eval: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load dataset and enrich training data with descriptor columns.

    Parameters
    ----------
    dataset_name : str
        Registry key (e.g., "solubility").
    use_tdc_eval : bool
        When True (default), loads TDC's fixed benchmark scaffold split and
        returns ``(train_val_df, test_df)``.  The test split is never biased.
        When False, loads the full dataset and returns ``(full_df, None)``
        so the caller handles splitting.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        (train_df, test_df).  test_df is None when use_tdc_eval=False.
    """
    if use_tdc_eval:
        from molgate.data.loaders import load_tdc_benchmark_split
        train_df, test_df = load_tdc_benchmark_split(dataset_name)
    else:
        from molgate.data.loaders import load_dataset
        train_df = load_dataset(dataset_name)
        test_df = None

    # Enrich training data with descriptor columns so bias functions that
    # filter on physicochemical properties (e.g. MW) can find them.
    train_df = _enrich_with_descriptors(train_df)

    n_test = len(test_df) if test_df is not None else "N/A"
    logger.info(
        f"Prepared {dataset_name}: {len(train_df)} train molecules, "
        f"{n_test} test molecules, {len(train_df.columns)} columns"
    )
    return train_df, test_df


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
    preloaded_df: pd.DataFrame | None = None,
    preloaded_test_df: pd.DataFrame | None = None,
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
        preloaded_df=preloaded_df,
        preloaded_test_df=preloaded_test_df,
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

@flow(name="bias_experiment", log_prints=True, task_runner=ConcurrentTaskRunner(max_workers=3))
def bias_experiment_flow(
    config_path: str | None = None,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    task_type: str = "regression",
    model_overrides: dict | None = None,
    wandb_mode: str = "disabled",
    use_tdc_eval: bool = True,
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
    use_tdc_eval : bool
        When True (default), use TDC's fixed benchmark scaffold split so
        evaluation is always on the same held-out test set regardless of
        which bias condition was applied to training data.  Results are
        directly comparable to TDC leaderboard submissions.
        When False, the full dataset is split per-run (original behaviour).

    Notes
    -----
    Parallelism is controlled by the flow's task runner, set to
    ``ConcurrentTaskRunner(max_workers=3)`` by default.  To override at
    call time use::

        bias_experiment_flow.with_options(
            task_runner=ConcurrentTaskRunner(max_workers=N)
        )(...)

    or pass ``--max-workers N`` to ``scripts/run_bias_experiment.py``.

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

    # 3. Pre-load each unique dataset once (with descriptor columns merged).
    #    Returns (train_df, test_df) in TDC eval mode, or (full_df, None)
    #    in legacy mode.  Bias is applied only to train_df; test_df is the
    #    fixed held-out set shared across all conditions.
    unique_datasets = list({r["dataset"] for r in matrix})
    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame | None]] = {}
    for ds in unique_datasets:
        logger.info(f"Preparing dataset: {ds}")
        prepared[ds] = task_prepare_dataset(ds, use_tdc_eval=use_tdc_eval)

    # 4. Submit all experiments concurrently.
    #    .submit() returns immediately with a PrefectFuture; the
    #    ConcurrentTaskRunner schedules up to max_workers at a time.
    logger.info(f"Submitting {len(matrix)} experiment runs...")
    futures = [
        task_run_single.submit(
            run_spec=run_spec,
            task_type=task_type,
            model_overrides=model_overrides,
            preloaded_df=prepared[run_spec["dataset"]][0],
            preloaded_test_df=prepared[run_spec["dataset"]][1],
            wandb_mode=wandb_mode,
        )
        for run_spec in matrix
    ]

    # 5. Collect results (blocks until all futures are done)
    results = [f.result() for f in futures]

    # 6. Aggregate into a comparison DataFrame
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
