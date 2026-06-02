"""W&B experiment tracking utilities.

Thin wrapper around the ``wandb`` SDK that standardises how molgate
initialises runs, logs metrics, and records artifacts.  Every training
run and Prefect flow calls these helpers instead of touching ``wandb``
directly — this keeps the W&B contract in one place and makes it easy
to switch to offline/disabled mode for CI or local development.

Design decisions:
    - **Config-driven**: ``init_run`` reads ``configs/wandb.yaml`` for
      project, entity, and mode so callers don't hard-code them.
    - **Tag conventions**: Tags always include dataset name, model type,
      and bias variant (if any).  The ``build_tags`` helper enforces this.
    - **Offline-first for tests**: ``mode="disabled"`` skips all network
      calls while still exercising the logging code paths.
    - **No global state**: Each function takes an explicit ``run`` object
      (or returns one).  No reliance on ``wandb.run`` global.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "wandb.yaml"


def load_wandb_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load W&B settings from the YAML config file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to ``wandb.yaml``.  Defaults to ``configs/wandb.yaml``
        relative to the project root.

    Returns
    -------
    dict
        Keys: ``project``, ``entity``, ``tags_prefix``, ``mode``.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(f"W&B config not found at {path}, using defaults")
        return {
            "project": "molgate",
            "entity": None,
            "tags_prefix": "molgate",
            "mode": "online",
        }

    with open(path) as f:
        raw = yaml.safe_load(f)

    return raw.get("wandb", raw)


# ---------------------------------------------------------------------------
# Tag building
# ---------------------------------------------------------------------------

def build_tags(
    dataset: str | None = None,
    model: str | None = None,
    bias: str | None = None,
    split: str | None = None,
    extra: list[str] | None = None,
    prefix: str | None = None,
) -> list[str]:
    """Build a consistent tag list for a W&B run.

    Tags follow the convention::

        ["molgate", "solubility", "lgbm_morgan", "bias:scaffold_top5", "split:scaffold"]

    Parameters
    ----------
    dataset : str, optional
        Dataset name (e.g., "solubility").
    model : str, optional
        Model name (e.g., "lgbm_morgan", "gnn").
    bias : str, optional
        Bias variant identifier (e.g., "scaffold_top10", "mw_narrow").
        Prefixed with "bias:" in the tag.  Use "unbiased" for no bias.
    split : str, optional
        Split strategy (e.g., "random", "scaffold").
        Prefixed with "split:" in the tag.
    extra : list[str], optional
        Additional free-form tags.
    prefix : str, optional
        Project-level prefix tag.  Defaults to the config's ``tags_prefix``.

    Returns
    -------
    list[str]
        Deduplicated, sorted tag list.
    """
    tags: list[str] = []

    if prefix:
        tags.append(prefix)

    if dataset:
        tags.append(dataset)

    if model:
        tags.append(model)

    if bias:
        tags.append(f"bias:{bias}")

    if split:
        tags.append(f"split:{split}")

    if extra:
        tags.extend(extra)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_tags = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)

    return unique_tags


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def init_run(
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    job_type: str | None = None,
    mode: str | None = None,
    wandb_config_path: str | Path | None = None,
    **kwargs: Any,
):
    """Initialise a W&B run with project-level defaults.

    Reads ``configs/wandb.yaml`` for project/entity/mode, then layers
    on the caller's overrides.  Returns the ``wandb.Run`` object.

    Parameters
    ----------
    name : str, optional
        Human-readable run name (e.g., "lgbm_morgan_solubility_unbiased").
    config : dict, optional
        Run config dict — hyperparameters, dataset info, etc.
        Logged to W&B and shown in the run's config panel.
    tags : list[str], optional
        Tags for filtering runs in the W&B dashboard.
    group : str, optional
        Group name for related runs (e.g., "bias_experiment_solubility").
    job_type : str, optional
        Job type (e.g., "train", "eda", "evaluate").
    mode : str, optional
        Override the config's mode.  "online", "offline", or "disabled".
    wandb_config_path : str or Path, optional
        Path to ``wandb.yaml``.
    **kwargs
        Additional keyword arguments passed to ``wandb.init()``.

    Returns
    -------
    wandb.Run
        The initialised W&B run.  Caller is responsible for calling
        ``run.finish()`` when done (or using it as a context manager).

    Examples
    --------
    >>> run = init_run(
    ...     name="lgbm_morgan_solubility",
    ...     config={"model": "lgbm_morgan", "dataset": "solubility", "lr": 0.05},
    ...     tags=build_tags(dataset="solubility", model="lgbm_morgan"),
    ...     job_type="train",
    ... )
    >>> run.log({"rmse": 1.05, "mae": 0.82})
    >>> run.finish()
    """
    import wandb

    wb_config = load_wandb_config(wandb_config_path)

    project = wb_config.get("project", "molgate")
    entity = wb_config.get("entity")  # None → default entity
    run_mode = mode or wb_config.get("mode", "online")

    # Prepend the project prefix tag if tags are provided
    prefix = wb_config.get("tags_prefix")
    if tags and prefix and prefix not in tags:
        tags = [prefix] + tags

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags,
        group=group,
        job_type=job_type,
        mode=run_mode,
        **kwargs,
    )

    logger.info(
        f"W&B run initialised: {run.name} (project={project}, "
        f"mode={run_mode}, tags={tags})"
    )

    return run


def log_metrics(run, metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics to a W&B run.

    Parameters
    ----------
    run : wandb.Run
        Active W&B run.
    metrics : dict[str, float]
        Metric name → value pairs.
    step : int, optional
        Step number (e.g., epoch).  If None, W&B auto-increments.
    """
    run.log(metrics, step=step)


def log_predictions_table(
    run,
    predictions_df,
    table_name: str = "predictions",
) -> None:
    """Log a predictions DataFrame as a W&B Table artifact.

    W&B Tables allow interactive filtering, sorting, and visualisation
    in the dashboard.  Each row is a molecule with SMILES, true value,
    predicted value, and error.

    Parameters
    ----------
    run : wandb.Run
        Active W&B run.
    predictions_df : pd.DataFrame
        Per-molecule predictions (from ``evaluate_model``).
    table_name : str
        Name for the table in the W&B run.
    """
    import wandb

    table = wandb.Table(dataframe=predictions_df)
    run.log({table_name: table})
    logger.info(f"Logged W&B Table '{table_name}' with {len(predictions_df)} rows")


def log_model_artifact(
    run,
    model_path: str | Path,
    artifact_name: str,
    artifact_type: str = "model",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a trained model file as a W&B Artifact.

    Artifacts are versioned, immutable blobs.  Useful for model
    checkpointing and lineage tracking.

    Parameters
    ----------
    run : wandb.Run
        Active W&B run.
    model_path : str or Path
        Path to the serialised model file (e.g., ``.pt`` or ``.joblib``).
    artifact_name : str
        Name for the artifact (e.g., "lgbm_morgan_solubility").
    artifact_type : str
        Artifact type (default "model").
    metadata : dict, optional
        Extra metadata to attach (e.g., metrics, hyperparameters).
    """
    import wandb

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    logger.info(f"Logged W&B artifact '{artifact_name}' from {model_path}")


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import numpy as np
    import pandas as pd

    # ===================================================================
    # Part 1: Config loading
    # ===================================================================
    print("=" * 60)
    print("Part 1: Load W&B config")
    print("=" * 60)

    config = load_wandb_config()
    print(f"  Project:     {config.get('project')}")
    print(f"  Entity:      {config.get('entity')}")
    print(f"  Mode:        {config.get('mode')}")
    print(f"  Tags prefix: {config.get('tags_prefix')}")

    # ===================================================================
    # Part 2: Tag building
    # ===================================================================
    print("\n" + "=" * 60)
    print("Part 2: Build tags")
    print("=" * 60)

    tags = build_tags(
        dataset="solubility",
        model="lgbm_morgan",
        bias="scaffold_top10",
        split="random",
        prefix="molgate",
    )
    print(f"  Tags: {tags}")

    tags_unbiased = build_tags(
        dataset="solubility",
        model="gnn",
        bias="unbiased",
        split="scaffold",
    )
    print(f"  Tags (unbiased): {tags_unbiased}")

    # ===================================================================
    # Part 3: Full run lifecycle (disabled mode — no network calls)
    # ===================================================================
    print("\n" + "=" * 60)
    print("Part 3: Full run lifecycle (disabled mode)")
    print("=" * 60)

    run = init_run(
        name="demo_lgbm_solubility",
        config={
            "model": "lgbm_morgan",
            "dataset": "solubility",
            "bias": "unbiased",
            "split": "random",
            "seed": 42,
            "n_estimators": 500,
            "learning_rate": 0.05,
        },
        tags=build_tags(
            dataset="solubility",
            model="lgbm_morgan",
            bias="unbiased",
        ),
        group="demo_experiment",
        job_type="train",
        mode="disabled",  # No network — safe for demo
    )

    # Simulate logging epoch metrics
    for epoch in range(5):
        log_metrics(run, {
            "train_loss": 2.0 - epoch * 0.3,
            "val_loss": 2.2 - epoch * 0.25,
            "val_rmse": 1.5 - epoch * 0.2,
        }, step=epoch)

    # Simulate final test metrics
    test_metrics = {"test_rmse": 1.10, "test_mae": 0.85, "test_r2": 0.78}
    log_metrics(run, test_metrics)

    # Simulate a predictions table
    pred_df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
        "y_true": [-0.77, -0.77, 0.17],
        "y_pred": [-0.85, -1.10, 0.22],
        "error": [-0.08, -0.33, 0.05],
        "abs_error": [0.08, 0.33, 0.05],
    })
    log_predictions_table(run, pred_df)

    # Log run summary
    run.summary.update(test_metrics)
    run.finish()

    print("\n  Run completed (disabled mode — no data sent to W&B)")
    print(f"  Run name: {run.name}")
    print(f"  Config logged: {dict(run.config)}")

    import IPython
    IPython.embed()
