"""Experiment runner: ExperimentSpec → train → evaluate → result JSON.

Each call to run_experiment() executes one full training run and writes
a result JSON to results_dir. run_experiment_grid() runs a list of specs,
optionally in parallel across processes.

W&B runs are initialized inside run_experiment() (not the caller) so
parallel workers each get their own run handle.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from admet.analysis.experiment import ExperimentSpec

logger = logging.getLogger(__name__)


def run_experiment(spec: ExperimentSpec) -> dict:
    """Execute one training run for one ExperimentSpec.

    Routes through load_benchmark_split() (fixed test set) or load_tdc_split()
    (dynamic splits) based on spec.data_source. Writes a result JSON to
    spec.results_dir and returns the same dict.

    Args:
        spec: Complete specification for this run.

    Returns:
        Result dict (also written to results_dir/{run_id}.json).
    """
    import dataclasses

    import torch

    from admet.dataset import (
        compute_split_statistics,
        load_benchmark_split,
        load_tdc_split,
    )
    from admet.evaluate import evaluate
    from admet.featurizer import get_featurizer
    from admet.model import ADMETModel
    from admet.train import train

    data_dir = Path(__file__).parent.parent.parent / "data"
    featurizer = get_featurizer("auto")

    if spec.data_source == "benchmark":
        train_ds, val_ds, test_ds = load_benchmark_split(
            benchmark_name=spec.tdc_dataset_name,
            split_type=spec.split_method,
            seed=spec.seed,
            data_dir=data_dir,
            featurizer=featurizer,
            task_type=spec.task_type,
            train_bias=spec.bias_config,
        )
    else:
        train_ds, val_ds, test_ds = load_tdc_split(
            tdc_dataset_name=spec.tdc_dataset_name,
            split_method=spec.split_method,
            seed=spec.seed,
            data_dir=data_dir,
            featurizer=featurizer,
            task_type=spec.task_type,
            train_bias=spec.bias_config,
        )

    split_stats = {
        "train": compute_split_statistics(train_ds._df, spec.task_type),
        "val": compute_split_statistics(val_ds._df, spec.task_type),
        "test": compute_split_statistics(test_ds._df, spec.task_type),
    }

    model_cfg = dataclasses.replace(
        spec.model_config,
        num_node_features=train_ds.num_node_features,
        num_edge_features=train_ds.num_edge_features,
        task_type=spec.task_type,
    )
    model = ADMETModel(model_cfg)

    wandb_run = None
    if spec.training_config.wandb_log:
        try:
            import wandb
            wandb_run = wandb.init(
                project=spec.wandb_project,
                group=spec.wandb_group,
                name=spec.run_id,
                config=spec.to_wandb_config(),
                tags=[
                    spec.property_name,
                    spec.split_method,
                    spec.bias_type,
                    spec.data_source,
                    "bias_study",
                ],
                reinit=True,
            )
        except Exception as e:
            logger.warning("W&B init failed (%s); continuing without logging.", e)

    start_time = time.time()

    ckpt_path = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=spec.training_config,
        split_stats=split_stats,
        tdc_dataset_name=spec.tdc_dataset_name,
        split_method=spec.split_method,
        wandb_run=wandb_run,
    )

    elapsed = time.time() - start_time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_metrics = evaluate(model, test_ds)

    train_stats = split_stats["train"]
    result: dict = {
        "run_id": spec.run_id,
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        "property": spec.property_name,
        "tdc_dataset_name": spec.tdc_dataset_name,
        "data_source": spec.data_source,
        "split_method": spec.split_method,
        "seed": spec.seed,
        "bias_type": spec.bias_type,
        "bias_params": (
            spec.bias_config.model_dump(exclude={"type"})
            if spec.bias_config is not None
            else {}
        ),
        "train_size": int(train_stats["n"]),
        "val_size": int(split_stats["val"]["n"]),
        "test_size": int(split_stats["test"]["n"]),
        "train_mw_mean": train_stats.get("mw_mean"),
        "train_mw_std": train_stats.get("mw_std"),
        "train_y_mean": train_stats.get("y_mean"),
        "train_y_std": train_stats.get("y_std"),
        "training_seconds": elapsed,
        "checkpoint_path": str(ckpt_path),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    if spec.task_type == "classification":
        result["train_pos_fraction"] = train_stats.get("pos_fraction")

    spec.results_dir.mkdir(parents=True, exist_ok=True)
    result_path = spec.results_dir / f"{spec.run_id}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Run complete: %s → %s", spec.run_id, result_path)

    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()

    return result


def run_experiment_grid(
    specs: list[ExperimentSpec],
    n_parallel: int = 1,
) -> list[dict]:
    """Run all experiments in the grid, sequentially or in parallel.

    Args:
        specs: List of ExperimentSpecs from expand_experiment_grid().
        n_parallel: Worker processes. Default 1 (sequential) to avoid GPU OOM.
            Set >1 for CPU-only runs or multi-GPU environments.

    Returns:
        List of result dicts in completion order.
    """
    results: list[dict] = []

    if n_parallel <= 1:
        for i, spec in enumerate(specs, 1):
            logger.info("Running %d/%d: %s", i, len(specs), spec.run_id)
            try:
                results.append(run_experiment(spec))
            except Exception as e:
                logger.error("Run %s failed: %s", spec.run_id, e, exc_info=True)
        return results

    with ProcessPoolExecutor(max_workers=n_parallel) as executor:
        futures = {executor.submit(run_experiment, spec): spec for spec in specs}
        completed = 0
        for future in as_completed(futures):
            spec = futures[future]
            completed += 1
            try:
                results.append(future.result())
                logger.info("Completed %d/%d: %s", completed, len(specs), spec.run_id)
            except Exception as e:
                logger.error("Run %s failed: %s", spec.run_id, e, exc_info=True)

    return results
