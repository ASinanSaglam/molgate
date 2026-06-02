"""GNN training loop with early stopping and LR scheduling.

The ``Trainer`` class handles the full training lifecycle for the GNN:
    1. Build optimizer (AdamW) and LR scheduler from config
    2. Run train/val epochs with gradient updates
    3. Track best validation metric and apply early stopping
    4. Return the best model state and training history

This trainer is GNN-specific — FingerprintModel uses scikit-learn's
``.fit()`` directly and doesn't need a custom training loop.

Design choices:
    - AdamW over Adam: AdamW decouples weight decay from the adaptive
      learning rate. With standard Adam, L2 regularisation interacts
      poorly with the per-parameter learning rate scaling. AdamW fixes
      this. For small models like ours the difference is minor, but
      AdamW is the modern default.
    - ReduceLROnPlateau over cosine/step schedulers: it adapts to the
      actual training dynamics rather than following a fixed schedule.
      If val loss plateaus, LR drops. If training is smooth, LR stays.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from molgate.training.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TrainHistory:
    """Records per-epoch metrics during training.

    Stored as lists where index = epoch number. The trainer appends
    one entry per epoch. Useful for plotting learning curves and
    diagnosing training issues (overfitting, divergence, etc.).
    """
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metrics: list[dict[str, float]] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    training_time_seconds: float = 0.0


class Trainer:
    """Training loop for MoleculeGNN.

    Parameters
    ----------
    model : MoleculeGNN
        The GNN model to train. Must already be on the correct device.
    task_type : str
        "regression" or "classification". Determines the loss function:
        MSELoss for regression, BCELoss for classification.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularisation strength. Applied via AdamW's decoupled
        weight decay (not as an L2 penalty on the loss).
    epochs : int
        Maximum number of training epochs.
    patience : int
        Early stopping patience. Training stops if validation loss
        doesn't improve for this many consecutive epochs.
    batch_size : int
        Number of molecular graphs per mini-batch.
    scheduler_patience : int
        Number of epochs with no val loss improvement before the
        LR scheduler reduces the learning rate.
    scheduler_factor : float
        Factor by which to reduce LR (new_lr = old_lr * factor).
    device : str
        "cpu" or "cuda". If "cuda" and no GPU is available, falls
        back to CPU with a warning.

    Examples
    --------
    >>> trainer = Trainer(model, task_type="regression")
    >>> history = trainer.fit(train_graphs, val_graphs)
    >>> print(f"Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}")
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "regression",
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 100,
        patience: int = 15,
        batch_size: int = 64,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.5,
        device: str = "cpu",
    ) -> None:
        # Device handling
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.task_type = task_type
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

        # Loss function
        if task_type == "regression":
            self.criterion = nn.MSELoss()
        elif task_type == "classification":
            # BCELoss expects sigmoid outputs — our model applies sigmoid
            # in forward() when task_type="classification"
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unknown task type: {task_type!r}")

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler: reduce LR when val loss plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",          # We want to minimise val loss
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

        logger.info(
            f"Trainer: task={task_type}, lr={lr}, wd={weight_decay}, "
            f"epochs={epochs}, patience={patience}, batch_size={batch_size}, "
            f"device={self.device}"
        )

    @classmethod
    def from_config(cls, model: nn.Module, training_config: dict, task_type: str = "regression"):
        """Create a Trainer from a config dict (e.g., from models.yaml).

        This is the typical way to create a Trainer — the config comes
        from ``model.training_config`` which was set by the factory.
        """
        return cls(
            model=model,
            task_type=task_type,
            lr=training_config.get("lr", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
            epochs=training_config.get("epochs", 100),
            patience=training_config.get("patience", 15),
            batch_size=training_config.get("batch_size", 64),
            scheduler_patience=training_config.get("scheduler_patience", 5),
            scheduler_factor=training_config.get("scheduler_factor", 0.5),
            device=training_config.get("device", "cpu"),
        )

    def fit(
        self,
        train_graphs: list,
        val_graphs: list,
    ) -> TrainHistory:
        """Train the model with early stopping.

        Parameters
        ----------
        train_graphs : list[Data]
            List of PyG Data objects for training. Each must have a ``y``
            attribute with the target value.
        val_graphs : list[Data]
            List of PyG Data objects for validation.

        Returns
        -------
        TrainHistory
            Training history with per-epoch losses, metrics, and LRs.
            The model's state is restored to the best epoch before
            returning.
        """
        # Build DataLoaders — PyG's DataLoader handles batching molecular
        # graphs by merging them into one disconnected graph with a
        # `batch` assignment vector.
        train_loader = DataLoader(
            train_graphs, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_graphs, batch_size=self.batch_size, shuffle=False
        )

        history = TrainHistory()
        best_state = None
        epochs_without_improvement = 0

        start_time = time.time()

        for epoch in range(self.epochs):
            # --- Train one epoch ---
            train_loss = self._train_epoch(train_loader)

            # --- Validate ---
            val_loss, val_preds, val_targets = self._validate(val_loader)

            # --- Compute validation metrics ---
            val_metrics = compute_metrics(val_targets, val_preds, self.task_type)

            # --- Record history ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_metrics.append(val_metrics)
            history.learning_rates.append(current_lr)

            # --- LR scheduler step ---
            self.scheduler.step(val_loss)

            # --- Early stopping check ---
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log every 10 epochs or on improvement
            if epoch % 10 == 0 or epochs_without_improvement == 0:
                primary_metric = "rmse" if self.task_type == "regression" else "auroc"
                logger.info(
                    f"Epoch {epoch:3d}/{self.epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"val_{primary_metric}={val_metrics.get(primary_metric, 0):.4f}, "
                    f"lr={current_lr:.6f}"
                    + (" *" if epochs_without_improvement == 0 else "")
                )

            if epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.patience} epochs). "
                    f"Best epoch: {history.best_epoch}"
                )
                break

        # Restore best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        history.training_time_seconds = time.time() - start_time
        logger.info(
            f"Training complete in {history.training_time_seconds:.1f}s. "
            f"Best val loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}"
        )

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch. Returns mean loss over all batches.

        The training loop for one epoch:
            1. Set model to train mode (enables dropout, BatchNorm uses
               batch statistics)
            2. For each mini-batch:
               a. Move data to device
               b. Forward pass → predictions
               c. Compute loss
               d. Backward pass → compute gradients
               e. Optimizer step → update weights
               f. Zero gradients for next batch
            3. Return average loss
        """
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in loader:
            batch = batch.to(self.device)

            # Forward pass
            preds = self.model(batch).squeeze(-1)  # (batch_size,)
            targets = batch.y.to(self.device)       # (batch_size,)

            # Compute loss
            loss = self.criterion(preds, targets)

            # Backward pass + update
            self.optimizer.zero_grad()  # Clear old gradients
            loss.backward()             # Compute new gradients
            self.optimizer.step()       # Update weights

            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        """Run validation. Returns (loss, predictions, targets).

        ``@torch.no_grad()`` disables gradient computation — we don't
        need gradients during validation, and skipping them saves memory
        and compute (~30% faster than with gradients enabled).

        ``.eval()`` switches the model to evaluation mode:
            - Dropout is disabled (all neurons active)
            - BatchNorm uses running statistics (computed during training)
              instead of batch statistics
        """
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            batch = batch.to(self.device)
            preds = self.model(batch).squeeze(-1)
            targets = batch.y.to(self.device)

            loss = self.criterion(preds, targets)
            total_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(n_samples, 1)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        return avg_loss, all_preds, all_targets

    @torch.no_grad()
    def predict(self, graphs: list) -> np.ndarray:
        """Run inference on a list of graphs. Returns predictions as numpy array."""
        self.model.eval()
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        for batch in loader:
            batch = batch.to(self.device)
            preds = self.model(batch).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.featurizer import smiles_list_to_graphs
    from molgate.models.gnn import MoleculeGNN

    from tdc import BenchmarkGroup

    # --- TDC 5-seed leaderboard evaluation ---
    # load_dataset() pulls the FULL TDC dataset. TDC's test set is a
    # fixed subset of that pool. If we random_split the full dataset,
    # ~73% of TDC test molecules leak into our training set, giving
    # artificially low MAE (~0.75). Using TDC's splits guarantees
    # zero overlap between train and test.
    #
    # TDC leaderboard requires 5 independent runs with different
    # train/val splits to report mean +/- std.

    BENCHMARK = "solubility_aqsoldb"
    SEEDS = [1, 2, 3, 4, 5]

    group = BenchmarkGroup(name="ADMET_Group", path="data/")
    predictions_list = []
    per_seed_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}/{len(SEEDS)}")
        print(f"{'='*60}")

        benchmark = group.get(BENCHMARK)
        test_df = benchmark["test"]

        train_df, val_df = group.get_train_valid_split(
            benchmark=BENCHMARK, split_type="default", seed=seed,
        )
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Featurize as graphs (TDC uses "Drug"/"Y" column names)
        train_graphs = smiles_list_to_graphs(
            train_df["Drug"].tolist(), train_df["Y"].tolist()
        )
        val_graphs = smiles_list_to_graphs(
            val_df["Drug"].tolist(), val_df["Y"].tolist()
        )
        test_graphs = smiles_list_to_graphs(
            test_df["Drug"].tolist(), test_df["Y"].tolist()
        )

        # Fresh model for each seed (no weight carryover)
        model = MoleculeGNN(hidden_dim=64, num_layers=2, task_type="regression")
        trainer = Trainer(
            model,
            task_type="regression",
            lr=0.001,
            epochs=100,
            patience=15,
            batch_size=64,
        )

        history = trainer.fit(train_graphs, val_graphs)
        print(f"  Best epoch: {history.best_epoch}, "
              f"best val loss: {history.best_val_loss:.4f}, "
              f"time: {history.training_time_seconds:.1f}s")

        # Predict on TDC test set
        test_preds = trainer.predict(test_graphs)

        # Single-seed TDC evaluation
        tdc_eval = group.evaluate(
            {BENCHMARK: test_preds}, benchmark=BENCHMARK
        )
        seed_mae = tdc_eval[BENCHMARK]["mae"]
        per_seed_results.append(seed_mae)
        print(f"  Seed {seed} MAE: {seed_mae:.4f}")

        predictions_list.append({BENCHMARK: test_preds})

    # --- Multi-seed summary (leaderboard format) ---
    results = group.evaluate_many(predictions_list)
    mean_mae, std_mae = results[BENCHMARK]

    print(f"\n{'='*60}")
    print(f"TDC Leaderboard Evaluation — {BENCHMARK}")
    print(f"{'='*60}")
    print(f"  Per-seed MAE: {[f'{m:.4f}' for m in per_seed_results]}")
    print(f"  Mean MAE: {mean_mae:.4f} +/- {std_mae:.4f}")
    print(f"  Model: GNN (hidden=64, layers=2, pool=mean)")
    print(f"  Seeds: {SEEDS}")

    import IPython
    IPython.embed()
