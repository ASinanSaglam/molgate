"""Tests for W&B integration (tracking.py + Trainer W&B logging).

All tests use wandb mode="disabled" — no network calls, no W&B account
needed.  This exercises the logging code paths without side effects.
"""

import numpy as np
import pandas as pd
import pytest

from molgate.tracking import (
    build_tags,
    init_run,
    load_wandb_config,
    log_metrics,
    log_predictions_table,
)


# ===================================================================
# Config loading
# ===================================================================


class TestLoadWandbConfig:
    """Tests for load_wandb_config."""

    def test_returns_dict(self):
        """Should return a dict with expected keys."""
        config = load_wandb_config()
        assert isinstance(config, dict)
        assert "project" in config
        assert "mode" in config

    def test_project_name(self):
        """Project should be 'molgate'."""
        config = load_wandb_config()
        assert config["project"] == "molgate"

    def test_fallback_on_missing_file(self):
        """Should return defaults when config file doesn't exist."""
        config = load_wandb_config("/nonexistent/path/wandb.yaml")
        assert config["project"] == "molgate"
        assert config["mode"] == "online"


# ===================================================================
# Tag building
# ===================================================================


class TestBuildTags:
    """Tests for build_tags."""

    def test_basic_tags(self):
        """Should include dataset, model, and bias tags."""
        tags = build_tags(dataset="solubility", model="lgbm_morgan", bias="unbiased")
        assert "solubility" in tags
        assert "lgbm_morgan" in tags
        assert "bias:unbiased" in tags

    def test_split_tag(self):
        """Split should be prefixed with 'split:'."""
        tags = build_tags(split="scaffold")
        assert "split:scaffold" in tags

    def test_prefix_tag(self):
        """Prefix should appear first."""
        tags = build_tags(prefix="molgate", dataset="solubility")
        assert tags[0] == "molgate"

    def test_no_duplicates(self):
        """Tags should be deduplicated."""
        tags = build_tags(
            prefix="molgate",
            dataset="solubility",
            extra=["molgate", "solubility"],
        )
        assert tags.count("molgate") == 1
        assert tags.count("solubility") == 1

    def test_empty_call(self):
        """Calling with no args should return empty list."""
        tags = build_tags()
        assert tags == []

    def test_extra_tags(self):
        """Extra tags should be appended."""
        tags = build_tags(extra=["custom_tag", "experiment_v2"])
        assert "custom_tag" in tags
        assert "experiment_v2" in tags


# ===================================================================
# Run lifecycle (disabled mode)
# ===================================================================


class TestInitRun:
    """Tests for init_run with mode='disabled'."""

    def test_returns_run_object(self):
        """Should return a wandb Run (or RunDisabled) object."""
        run = init_run(name="test_run", mode="disabled")
        try:
            assert run is not None
        finally:
            run.finish()

    def test_config_logged(self):
        """Config dict should be accessible on the run."""
        run = init_run(
            name="test_config",
            config={"lr": 0.01, "dataset": "solubility"},
            mode="disabled",
        )
        try:
            # In disabled mode, config is still stored
            assert run.config is not None
        finally:
            run.finish()

    def test_tags_passed(self):
        """Tags should be passed to the run."""
        tags = ["solubility", "lgbm_morgan"]
        run = init_run(name="test_tags", tags=tags, mode="disabled")
        try:
            assert run is not None
        finally:
            run.finish()


class TestLogMetrics:
    """Tests for log_metrics in disabled mode."""

    def test_log_metrics_no_error(self):
        """Logging metrics in disabled mode should not raise."""
        run = init_run(name="test_log", mode="disabled")
        try:
            log_metrics(run, {"rmse": 1.05, "mae": 0.82}, step=0)
            log_metrics(run, {"rmse": 0.95, "mae": 0.75}, step=1)
        finally:
            run.finish()


class TestLogPredictionsTable:
    """Tests for log_predictions_table in disabled mode."""

    def test_log_table_no_error(self):
        """Logging a predictions table in disabled mode should not raise."""
        run = init_run(name="test_table", mode="disabled")
        try:
            df = pd.DataFrame({
                "smiles": ["CCO", "c1ccccc1"],
                "y_true": [-0.77, -0.77],
                "y_pred": [-0.85, -1.10],
                "error": [-0.08, -0.33],
            })
            log_predictions_table(run, df)
        finally:
            run.finish()


# ===================================================================
# Trainer + W&B integration
# ===================================================================


class TestTrainerWandbIntegration:
    """Tests for Trainer with wandb_run parameter."""

    def test_trainer_with_wandb_disabled(self):
        """Trainer should log to W&B in disabled mode without errors."""
        from molgate.data.featurizer import smiles_list_to_graphs
        from molgate.models.gnn import MoleculeGNN
        from molgate.training.trainer import Trainer

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O", "CCCO",
                   "c1ccncc1", "C1CCCCC1", "c1ccc(O)cc1"]
        targets = [-0.77, -0.77, 0.17, -0.5, -1.0, -2.0, -4.0, -0.5]

        graphs = smiles_list_to_graphs(smiles, targets)
        train_graphs = graphs[:5]
        val_graphs = graphs[5:]

        run = init_run(
            name="test_trainer_wandb",
            config={"model": "gnn", "dataset": "test"},
            mode="disabled",
        )
        try:
            model = MoleculeGNN(hidden_dim=16, num_layers=1, task_type="regression")
            trainer = Trainer(
                model, task_type="regression",
                lr=0.01, epochs=3, patience=10, batch_size=4,
                wandb_run=run,
            )

            history = trainer.fit(train_graphs, val_graphs)

            # Verify training still works correctly with W&B attached
            assert len(history.train_loss) == 3
            assert len(history.val_loss) == 3
        finally:
            run.finish()

    def test_trainer_without_wandb(self):
        """Trainer should work normally when wandb_run is None (default)."""
        from molgate.data.featurizer import smiles_list_to_graphs
        from molgate.models.gnn import MoleculeGNN
        from molgate.training.trainer import Trainer

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O", "CCCO",
                   "c1ccncc1", "C1CCCCC1", "c1ccc(O)cc1"]
        targets = [-0.77, -0.77, 0.17, -0.5, -1.0, -2.0, -4.0, -0.5]

        graphs = smiles_list_to_graphs(smiles, targets)
        train_graphs = graphs[:5]
        val_graphs = graphs[5:]

        model = MoleculeGNN(hidden_dim=16, num_layers=1, task_type="regression")
        trainer = Trainer(
            model, task_type="regression",
            lr=0.01, epochs=3, patience=10, batch_size=4,
            # No wandb_run — default None
        )

        history = trainer.fit(train_graphs, val_graphs)
        assert len(history.train_loss) == 3


class TestFullWandbWorkflow:
    """End-to-end test: init → train → evaluate → log → finish."""

    def test_full_workflow_disabled(self):
        """Full workflow in disabled mode should complete without errors."""
        from molgate.data.featurizer import smiles_list_to_graphs
        from molgate.models.gnn import MoleculeGNN
        from molgate.training.evaluate import evaluate_model
        from molgate.training.trainer import Trainer

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O", "CCCO",
                   "c1ccncc1", "C1CCCCC1", "c1ccc(O)cc1"]
        targets = [-0.77, -0.77, 0.17, -0.5, -1.0, -2.0, -4.0, -0.5]

        graphs = smiles_list_to_graphs(smiles, targets)
        train_graphs = graphs[:5]
        val_graphs = graphs[5:]
        test_graphs = graphs[5:]
        test_y = np.array(targets[5:])

        # 1. Init W&B run
        run = init_run(
            name="full_workflow_test",
            config={
                "model": "gnn",
                "dataset": "test",
                "hidden_dim": 16,
                "num_layers": 1,
            },
            tags=build_tags(dataset="test", model="gnn", bias="unbiased"),
            job_type="train",
            mode="disabled",
        )

        try:
            # 2. Train with W&B logging
            model = MoleculeGNN(hidden_dim=16, num_layers=1, task_type="regression")
            trainer = Trainer(
                model, task_type="regression",
                lr=0.01, epochs=3, patience=10, batch_size=4,
                wandb_run=run,
            )
            history = trainer.fit(train_graphs, val_graphs)

            # 3. Evaluate
            metrics, pred_df = evaluate_model(
                model=trainer, test_data=test_graphs,
                y_true=test_y, task_type="regression",
                smiles=smiles[5:],
            )

            # 4. Log test metrics and predictions table
            log_metrics(run, {f"test_{k}": v for k, v in metrics.items()})
            log_predictions_table(run, pred_df)

            # 5. Update summary
            run.summary.update({f"test_{k}": v for k, v in metrics.items()})

            # Verify everything worked
            assert len(history.train_loss) == 3
            assert isinstance(metrics, dict)
            assert "rmse" in metrics
            assert len(pred_df) == len(test_graphs)

        finally:
            run.finish()
