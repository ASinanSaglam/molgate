"""CLI for running a full experiment grid from a YAML config.

Usage:
    python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml
    python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --dry-run
    python scripts/run_bias_study.py configs/experiments/herg_bias_study.yaml --n-parallel 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a bias study experiment grid.")
    p.add_argument("config", type=Path, help="Path to experiment YAML config.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded experiment grid without running anything.",
    )
    p.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 sequential).",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging for all runs in this grid.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N specs from the grid (useful for testing).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from omegaconf import OmegaConf

    from admet.analysis.experiment import expand_experiment_grid
    from admet.analysis.report import save_tables
    from admet.analysis.runner import run_experiment_grid

    cfg = OmegaConf.load(args.config)
    specs = expand_experiment_grid(cfg)

    if args.no_wandb:
        from dataclasses import replace
        specs = [
            replace(s, training_config=replace(s.training_config, wandb_log=False))
            for s in specs
        ]

    if args.limit is not None:
        specs = specs[: args.limit]

    total = len(specs)
    logger.info(
        "Experiment grid: %d runs (%d seeds × %d splits × %d bias variants)",
        total,
        len(set(s.seed for s in specs)),
        len(set(s.split_method for s in specs)),
        len(set(s.bias_type for s in specs)),
    )

    if args.dry_run:
        print(f"\nDry run — {total} specs (not executing):\n")
        for i, spec in enumerate(specs, 1):
            bias_str = (
                spec.bias_config.model_dump_json()
                if spec.bias_config is not None
                else "no_bias"
            )
            print(f"  {i:3d}. {spec.run_id}  bias={bias_str}")
        return

    results = run_experiment_grid(specs, n_parallel=args.n_parallel)

    success = len(results)
    failed = total - success
    logger.info("Grid complete: %d succeeded, %d failed.", success, failed)

    # Auto-generate summary tables after the grid completes
    if results:
        results_dir = specs[0].results_dir
        try:
            save_tables(results_dir)
        except Exception as e:
            logger.warning("Failed to generate summary tables: %s", e)


if __name__ == "__main__":
    main()
