"""CLI for training a single ADMET property model.

Usage:
    # Benchmark mode (default) — fixed test set, leaderboard-comparable
    python scripts/train_property.py --property solubility --split scaffold --seed 42

    # Exploration mode — dynamic splits via single_pred
    python scripts/train_property.py --property solubility --split random --data-source single_pred

    # No W&B (dry run / testing)
    python scripts/train_property.py --property herg --split scaffold --no-wandb
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROPERTIES_CONFIG = REPO_ROOT / "configs" / "properties.yaml"
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one ADMET property model.")
    p.add_argument("--property", required=True, help="Property name (key in properties.yaml)")
    p.add_argument(
        "--split",
        default=None,
        choices=["random", "scaffold"],
        help="Split strategy. Defaults to the property's default_split.",
    )
    p.add_argument(
        "--data-source",
        default="benchmark",
        choices=["benchmark", "single_pred"],
        help=(
            "benchmark (default): fixed test set via admet_group, leaderboard-comparable. "
            "single_pred: dynamic splits, for exploration only."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="molgate-production")
    p.add_argument("--featurizer", default="auto", choices=["auto", "deepchem", "rdkit"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(PROPERTIES_CONFIG) as f:
        props = yaml.safe_load(f)["properties"]

    if args.property not in props:
        logger.error("Unknown property '%s'. Available: %s", args.property, list(props))
        sys.exit(1)

    prop_cfg = props[args.property]
    split_method = args.split or prop_cfg["default_split"]
    task_type = prop_cfg["task_type"]
    tdc_name = prop_cfg["tdc_name"]

    from admet.dataset import (
        compute_split_statistics,
        load_benchmark_split,
        load_tdc_split,
    )
    from admet.evaluate import evaluate
    from admet.featurizer import get_featurizer
    from admet.model import ADMETModel, ModelConfig
    from admet.train import TrainingConfig, train

    featurizer = get_featurizer(args.featurizer)

    if args.data_source == "benchmark":
        train_ds, val_ds, test_ds = load_benchmark_split(
            benchmark_name=tdc_name,
            split_type=split_method,
            seed=args.seed,
            data_dir=DATA_DIR,
            featurizer=featurizer,
            task_type=task_type,
        )
    else:
        train_ds, val_ds, test_ds = load_tdc_split(
            tdc_dataset_name=tdc_name,
            split_method=split_method,
            seed=args.seed,
            data_dir=DATA_DIR,
            featurizer=featurizer,
            task_type=task_type,
        )

    split_stats = {
        "train": compute_split_statistics(train_ds._df, task_type),
        "val": compute_split_statistics(val_ds._df, task_type),
        "test": compute_split_statistics(test_ds._df, task_type),
    }

    logger.info(
        "Loaded %s [%s]: train=%d val=%d test=%d | split=%s",
        tdc_name, args.data_source, len(train_ds), len(val_ds), len(test_ds), split_method,
    )

    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_node_features=train_ds.num_node_features,
        num_edge_features=train_ds.num_edge_features,
        task_type=task_type,
    )
    model = ADMETModel(model_cfg)

    train_cfg = TrainingConfig(
        epochs=args.epochs or 100,
        lr=args.lr or 1e-3,
        batch_size=args.batch_size or 64,
        checkpoint_dir=CHECKPOINT_DIR / args.property,
        seed=args.seed,
        wandb_log=not args.no_wandb,
    )

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"{args.property}_{split_method}_seed{args.seed}",
            config={
                "property": args.property,
                "tdc_dataset_name": tdc_name,
                "data_source": args.data_source,
                "task_type": task_type,
                "model_hidden_dim": model_cfg.hidden_dim,
                "model_num_layers": model_cfg.num_layers,
            },
            tags=[args.property, split_method, task_type, args.data_source],
        )

    ckpt_path = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        cfg=train_cfg,
        split_stats=split_stats,
        tdc_dataset_name=tdc_name,
        split_method=split_method,
        wandb_run=wandb_run,
    )

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_metrics = evaluate(model, test_ds)

    logger.info("Test metrics: %s", test_metrics)

    if wandb_run is not None:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.finish()

    logger.info("Checkpoint saved to: %s", ckpt_path)
    print(f"\nTest metrics for {args.property} ({split_method}, {args.data_source}, seed {args.seed}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
