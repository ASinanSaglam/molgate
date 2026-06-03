"""Run the bias experiment matrix and save results.

Usage
-----
# LightGBM only (fast, ~10-15 min, 3 concurrent runs):
python scripts/run_bias_experiment.py

# Full matrix including GNN (reduce concurrency to avoid OOM):
python scripts/run_bias_experiment.py --models lgbm_morgan lgbm_descriptors gnn --max-workers 1

# Specific conditions only:
python scripts/run_bias_experiment.py --conditions unbiased scaffold_top10 mw_narrow

# Offline W&B (sync later with `wandb sync`):
python scripts/run_bias_experiment.py --wandb-mode offline

# Override parallelism:
python scripts/run_bias_experiment.py --max-workers 6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run molgate bias experiment matrix.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["solubility"],
        help="Datasets to run (default: solubility).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lgbm_morgan", "lgbm_descriptors"],
        help="Models to include (default: lgbm_morgan lgbm_descriptors).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Bias conditions to run (default: all conditions in config).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to use (default: from bias_experiments.yaml).",
    )
    parser.add_argument(
        "--task-type",
        default="regression",
        choices=["regression", "classification"],
        help="Task type (default: regression).",
    )
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (default: online).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Max concurrent experiment runs (default: 3). Use 1 for GNN to avoid OOM.",
    )
    parser.add_argument(
        "--no-tdc-eval",
        action="store_true",
        help=(
            "Disable TDC benchmark evaluation. By default the fixed TDC "
            "scaffold split is used so results are leaderboard-comparable. "
            "Pass this flag to use per-run custom splits instead."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results CSV (default: outputs/<dataset>_<models>_results.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from prefect.task_runners import ConcurrentTaskRunner
    from molgate.flows.bias_experiment import bias_experiment_flow

    model_tag = "_".join(args.models)
    dataset_tag = "_".join(args.datasets)

    print("=" * 60)
    print("Bias Experiment Run")
    print("=" * 60)
    print(f"  Datasets    : {args.datasets}")
    print(f"  Models      : {args.models}")
    print(f"  Conditions  : {args.conditions or 'all'}")
    print(f"  Seeds       : {args.seeds or 'from config'}")
    print(f"  Task type   : {args.task_type}")
    print(f"  W&B mode    : {args.wandb_mode}")
    print(f"  Max workers : {args.max_workers}")
    print(f"  TDC eval    : {not args.no_tdc_eval}")
    print("=" * 60)

    flow = bias_experiment_flow.with_options(
        task_runner=ConcurrentTaskRunner(max_workers=args.max_workers)
    )

    results = flow(
        datasets=args.datasets,
        models=args.models,
        conditions=args.conditions,
        seeds=args.seeds,
        task_type=args.task_type,
        wandb_mode=args.wandb_mode,
        use_tdc_eval=not args.no_tdc_eval,
    )

    # Save results
    out_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / f"{dataset_tag}_{model_tag}_results.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    # Print summary table
    metric_cols = [c for c in ("rmse", "mae", "r2", "auroc") if c in results.columns]
    display_cols = ["model", "condition", "seed"] + metric_cols
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(results[display_cols].to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
