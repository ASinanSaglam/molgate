"""Training loop with W&B logging and model checkpointing.

Seeds torch, numpy, and random at the start of each run. All seeding is
logged to W&B so runs are fully reproducible from the config alone.

Checkpoint format (saved with torch.save as a dict):
    - model_state_dict: model weights
    - model_config: ModelConfig dataclass
    - training_config: TrainingConfig dataclass
    - val_metric: best validation metric value
    - val_metric_name: metric name string
    - tdc_dataset_name: str
    - split_method: str
    - epoch: int (epoch at which the best val metric was achieved)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from admet.dataset import ADMETDataset, TaskType
from admet.model import ADMETModel, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and bookkeeping settings for one training run.

    Args:
        epochs: Maximum number of training epochs.
        lr: Initial learning rate for Adam.
        batch_size: Number of graphs per mini-batch.
        patience: Early stopping patience (epochs without val improvement).
        checkpoint_dir: Directory for saving model checkpoints.
        seed: Random seed (seeded into torch, numpy, and random).
        wandb_log: If True, log metrics to W&B. Set False for dry runs.
    """

    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 64
    patience: int = 15
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    seed: int = 42
    wandb_log: bool = True


def seed_everything(seed: int) -> None:
    """Seed torch, numpy, and random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    model: ADMETModel,
    train_dataset: ADMETDataset,
    val_dataset: ADMETDataset,
    cfg: TrainingConfig,
    split_stats: dict[str, dict[str, float]],
    tdc_dataset_name: str,
    split_method: str,
    wandb_run: object | None = None,
) -> Path:
    """Train model with early stopping, log to W&B, save best checkpoint.

    Args:
        model: ADMETModel to train (moved to available device internally).
        train_dataset: Featurized training split.
        val_dataset: Featurized validation split.
        cfg: Training hyperparameters.
        split_stats: Pre-computed dataset statistics for W&B config logging.
            Format: {"train": {...}, "val": {...}, "test": {...}}.
            Computed by compute_split_statistics() in dataset.py.
        tdc_dataset_name: TDC dataset name, logged to checkpoint.
        split_method: Split strategy used, logged to checkpoint.
        wandb_run: Active W&B Run object, or None to skip W&B logging.

    Returns:
        Path to the saved checkpoint file.
    """
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    task_type = train_dataset.task_type
    loss_fn = _get_loss_fn(task_type)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=cfg.patience // 2, factor=0.5
    )

    # Log config and dataset stats to W&B
    if wandb_run is not None and cfg.wandb_log:
        flat_stats = {
            f"{split}_{k}": v
            for split, kv in split_stats.items()
            for k, v in kv.items()
        }
        wandb_run.config.update({
            "tdc_dataset_name": tdc_dataset_name,
            "split_method": split_method,
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "patience": cfg.patience,
            "task_type": task_type,
            **flat_stats,
        })

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_checkpoint_path: Path | None = None

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        val_loss = _run_epoch(model, val_loader, loss_fn, None, device, train=False)
        scheduler.step(val_loss)

        if wandb_run is not None and cfg.wandb_log:
            wandb_run.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            best_checkpoint_path = _save_checkpoint(
                model, cfg, tdc_dataset_name, split_method, best_val_loss, epoch
            )
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, cfg.patience)
                break

        if epoch % 10 == 0:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val_loss=%.4f | best=%.4f (ep %d)",
                epoch, train_loss, val_loss, best_val_loss, best_epoch,
            )

    elapsed = time.time() - start_time
    logger.info(
        "Training complete in %.1fs. Best val_loss=%.4f at epoch %d.",
        elapsed, best_val_loss, best_epoch,
    )

    if best_checkpoint_path is None:
        # Edge case: no checkpoint saved (shouldn't happen in practice)
        best_checkpoint_path = _save_checkpoint(
            model, cfg, tdc_dataset_name, split_method, best_val_loss, epoch
        )

    return best_checkpoint_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_loss_fn(task_type: TaskType) -> nn.Module:
    if task_type == "regression":
        return nn.MSELoss()
    return nn.BCEWithLogitsLoss()


def _run_epoch(
    model: ADMETModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    n_graphs = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch)
            loss = loss_fn(preds, batch.y)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

    return total_loss / n_graphs if n_graphs > 0 else 0.0


def _save_checkpoint(
    model: ADMETModel,
    cfg: TrainingConfig,
    tdc_dataset_name: str,
    split_method: str,
    val_metric: float,
    epoch: int,
) -> Path:
    fname = f"{tdc_dataset_name}_{split_method}_seed{cfg.seed}_ep{epoch}.pt"
    path = cfg.checkpoint_dir / fname
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.cfg,
            "training_config": cfg,
            "val_metric": val_metric,
            "val_metric_name": "val_loss",
            "tdc_dataset_name": tdc_dataset_name,
            "split_method": split_method,
            "epoch": epoch,
        },
        path,
    )
    return path


def load_checkpoint(path: Path) -> tuple[ADMETModel, dict]:
    """Load a saved checkpoint and reconstruct the model.

    Returns:
        (model, checkpoint_dict) where model is in eval mode.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ADMETModel(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt
