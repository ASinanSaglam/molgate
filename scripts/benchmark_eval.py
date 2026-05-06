"""TDC ADMET leaderboard evaluation script.

Trains one model per dataset across N seeds using the fixed admet_group splits,
collects aligned predictions on the fixed test sets, and calls bg.evaluate_many()
to produce leaderboard-format results ([mean, std] per dataset per metric).

The baseline runs (no bias, scaffold split) from run_bias_study.py are the
inputs to this script. Biased model predictions can be evaluated separately
to produce a "biased leaderboard" table for the analysis section.

Usage:
    # Full leaderboard run — all 22 datasets, 5 seeds (long)
    python scripts/benchmark_eval.py --seeds 42 123 456 789 1337

    # Quick validation — one dataset, 2 seeds
    python scripts/benchmark_eval.py --datasets solubility_aqsoldb herg --seeds 42 123 --no-wandb

    # Evaluate from saved checkpoints (skip training)
    python scripts/benchmark_eval.py --from-checkpoints results/benchmark_checkpoints/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "benchmark"

# All 22 admet_group datasets in evaluation order
ALL_BENCHMARK_DATASETS = [
    "caco2_wang",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "lipophilicity_astrazeneca",
    "solubility_aqsoldb",
    "bbb_martins",
    "ppbr_az",
    "vdss_lombardo",
    "cyp2d6_veith",
    "cyp3a4_veith",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp3a4_substrate_carbonmangels",
    "cyp2c9_substrate_carbonmangels",
    "half_life_obach",
    "clearance_microsome_az",
    "clearance_hepatocyte_az",
    "herg",
    "ames",
    "dili",
    "ld50_zhu",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TDC ADMET leaderboard evaluation.")
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1337],
        help="Random seeds. TDC leaderboard requires ≥5.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of datasets to evaluate. Defaults to all 22.",
    )
    p.add_argument(
        "--split",
        default="scaffold",
        choices=["scaffold", "random"],
        help="Split strategy for train/val partition (test is always fixed).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="molgate-leaderboard")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "benchmark",
        help="Directory for result JSONs and leaderboard output.",
    )
    return p.parse_args()


def train_one(
    dataset_name: str,
    task_type: str,
    split: str,
    seed: int,
    epochs: int,
    hidden_dim: int,
    num_layers: int,
    batch_size: int,
    patience: int,
    wandb_log: bool,
    wandb_project: str,
    output_dir: Path,
) -> tuple[dict, "np.ndarray"]:  # type: ignore[name-defined]
    """Train one model and return (metrics_dict, aligned_test_predictions)."""
    import dataclasses

    import numpy as np
    import torch

    from admet.dataset import BENCHMARK_TASK_TYPES, compute_split_statistics, load_benchmark_split
    from admet.evaluate import evaluate, predict_aligned
    from admet.featurizer import get_featurizer
    from admet.model import ADMETModel, ModelConfig
    from admet.train import TrainingConfig, train

    featurizer = get_featurizer("auto")

    train_ds, val_ds, test_ds = load_benchmark_split(
        benchmark_name=dataset_name,
        split_type=split,
        seed=seed,
        data_dir=DATA_DIR,
        featurizer=featurizer,
        task_type=task_type,
        train_bias=None,  # always no bias for leaderboard baseline
    )

    split_stats = {
        "train": compute_split_statistics(train_ds._df, task_type),
        "val":   compute_split_statistics(val_ds._df, task_type),
        "test":  compute_split_statistics(test_ds._df, task_type),
    }

    model_cfg = ModelConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_node_features=train_ds.num_node_features,
        num_edge_features=train_ds.num_edge_features,
        task_type=task_type,
    )
    model = ADMETModel(model_cfg)

    ckpt_subdir = CHECKPOINT_DIR / dataset_name
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        checkpoint_dir=ckpt_subdir,
        seed=seed,
        wandb_log=wandb_log,
    )

    wandb_run = None
    if wandb_log:
        try:
            import wandb
            wandb_run = wandb.init(
                project=wandb_project,
                group="leaderboard_baseline",
                name=f"{dataset_name}_{split}_seed{seed}",
                config={
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "data_source": "benchmark",
                    "split": split,
                    "seed": seed,
                    "bias": "none",
                },
                tags=[dataset_name, split, "leaderboard", "no_bias"],
                reinit=True,
            )
        except Exception as e:
            logger.warning("W&B init failed: %s", e)

    ckpt_path = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=train_cfg,
        split_stats=split_stats,
        tdc_dataset_name=dataset_name,
        split_method=split,
        wandb_run=wandb_run,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_metrics = evaluate(model, test_ds)
    # Aligned predictions for leaderboard submission (one value per test molecule)
    aligned_preds = predict_aligned(model, test_ds)

    logger.info(
        "[%s seed=%d] test metrics: %s",
        dataset_name, seed, {k: f"{v:.4f}" for k, v in test_metrics.items()},
    )

    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()

    return test_metrics, aligned_preds


def main() -> None:
    args = parse_args()

    from admet.dataset import BENCHMARK_TASK_TYPES
    from tdc.benchmark_group import admet_group

    datasets = args.datasets or ALL_BENCHMARK_DATASETS
    unknown = [d for d in datasets if d not in BENCHMARK_TASK_TYPES]
    if unknown:
        logger.error("Unknown datasets: %s", unknown)
        sys.exit(1)

    if len(args.seeds) < 5:
        logger.warning(
            "TDC leaderboard requires ≥5 seeds for submission; got %d. "
            "Results will not be officially submittable.",
            len(args.seeds),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # all_seed_preds[seed_idx] = {dataset_name: np.ndarray of predictions}
    all_seed_preds: list[dict] = []
    # per_run_metrics[dataset_name] = list of metric dicts (one per seed)
    per_run_metrics: dict[str, list[dict]] = {d: [] for d in datasets}

    for seed in args.seeds:
        logger.info("=== Seed %d ===", seed)
        seed_preds: dict = {}

        for dataset_name in datasets:
            task_type = BENCHMARK_TASK_TYPES[dataset_name]
            logger.info("  Training %s (%s)...", dataset_name, task_type)

            try:
                metrics, aligned_preds = train_one(
                    dataset_name=dataset_name,
                    task_type=task_type,
                    split=args.split,
                    seed=seed,
                    epochs=args.epochs,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    wandb_log=not args.no_wandb,
                    wandb_project=args.wandb_project,
                    output_dir=args.output_dir,
                )
                seed_preds[dataset_name] = aligned_preds
                per_run_metrics[dataset_name].append(metrics)
            except Exception as e:
                logger.error("  FAILED %s seed=%d: %s", dataset_name, seed, e, exc_info=True)

        all_seed_preds.append(seed_preds)

    # Leaderboard evaluation via TDC's evaluate_many
    logger.info("Computing leaderboard results across %d seeds...", len(args.seeds))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bg = admet_group(path=str(DATA_DIR))

    # Only pass seeds that completed all datasets
    complete_preds = [p for p in all_seed_preds if len(p) == len(datasets)]
    if complete_preds:
        try:
            leaderboard_results = bg.evaluate_many(complete_preds)
            lb_path = args.output_dir / "leaderboard_results.json"
            with open(lb_path, "w") as f:
                # evaluate_many returns {dataset: [mean, std]}
                json.dump(leaderboard_results, f, indent=2)
            logger.info("Leaderboard results written to %s", lb_path)

            print("\n=== TDC Leaderboard Results ===")
            print(f"{'Dataset':<40} {'Mean':>10} {'Std':>10}")
            print("-" * 62)
            for dataset, (mean, std) in sorted(leaderboard_results.items()):
                print(f"{dataset:<40} {mean:>10.4f} {std:>10.4f}")
        except Exception as e:
            logger.error("bg.evaluate_many() failed: %s", e, exc_info=True)
    else:
        logger.warning("No complete seed runs — skipping evaluate_many.")

    # Per-dataset summary across seeds (our own aggregation)
    summary: dict[str, dict] = {}
    for dataset_name, metrics_list in per_run_metrics.items():
        if not metrics_list:
            continue
        import numpy as np
        summary[dataset_name] = {
            k: {
                "mean": float(np.mean([m[k] for m in metrics_list])),
                "std": float(np.std([m[k] for m in metrics_list])),
            }
            for k in metrics_list[0]
        }

    summary_path = args.output_dir / "per_dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Per-dataset summary written to %s", summary_path)


if __name__ == "__main__":
    main()
