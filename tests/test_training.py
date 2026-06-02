"""Tests for training/metrics.py, training/trainer.py, and training/evaluate.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from molgate.data.featurizer import smiles_to_graph, smiles_list_to_graphs
from molgate.models.baseline import FingerprintModel
from molgate.models.gnn import MoleculeGNN
from molgate.training.metrics import compute_metrics
from molgate.training.trainer import Trainer, TrainHistory
from molgate.training.evaluate import evaluate_model, evaluate_tdc, evaluate_tdc_multi_seed

# Same data as conftest — duplicated to avoid import issues.
SAMPLE_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "c1ccc(O)cc1", "CC(=O)O",
    "c1ccncc1", "C1CCCCC1", "c1ccc(-c2ccccc2)cc1", "CC(C)(C)c1ccc(O)cc1",
]
SAMPLE_TARGETS = np.array([-1.0, -2.5, -1.5, -3.0, -0.5, 0.5, -2.0, -4.0, -3.5, -1.8])

# ---------------------------------------------------------------------------
# Helpers — tiny graph datasets for Trainer / evaluate tests
# ---------------------------------------------------------------------------

_TINY_SMILES = SAMPLE_SMILES[:8]
_TINY_TARGETS = SAMPLE_TARGETS[:8].tolist()


def _make_graph_dataset(smiles: list[str], targets: list[float]):
    """Build a list of PyG Data objects from SMILES + targets."""
    return smiles_list_to_graphs(smiles, targets)


def _make_tiny_gnn(task_type: str = "regression") -> MoleculeGNN:
    """Create a very small GNN for fast tests."""
    return MoleculeGNN(
        hidden_dim=16, num_layers=1, task_type=task_type, dropout=0.0, pool="mean",
    )


# ===================================================================
# Metrics (training/metrics.py)
# ===================================================================


class TestComputeMetricsRegression:
    """Tests for compute_metrics with task_type='regression'."""

    def test_returns_expected_keys(self):
        """Regression metrics dict should have rmse, mae, r2, pearson_r."""
        m = compute_metrics([1, 2, 3], [1.1, 2.1, 2.9], "regression")
        assert set(m.keys()) == {"rmse", "mae", "r2", "pearson_r"}

    def test_perfect_predictions(self):
        """Perfect predictions should give RMSE=0, R2=1."""
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        m = compute_metrics(y, y, "regression")
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2"] == pytest.approx(1.0, abs=1e-10)
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)
        assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)

    def test_constant_predictions(self):
        """Constant predictions: R2 should be <= 0, pearson_r should be NaN."""
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [3.0, 3.0, 3.0, 3.0, 3.0]
        m = compute_metrics(y_true, y_pred, "regression")
        assert m["r2"] <= 0.0
        assert np.isnan(m["pearson_r"])

    def test_all_values_float(self):
        """All metric values should be Python floats."""
        m = compute_metrics([1, 2, 3], [1.5, 2.5, 3.5], "regression")
        for v in m.values():
            assert isinstance(v, float)


class TestComputeMetricsClassification:
    """Tests for compute_metrics with task_type='classification'."""

    def test_returns_expected_keys(self):
        """Classification metrics dict should have auroc, accuracy, f1, precision, recall."""
        m = compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3], "classification")
        assert set(m.keys()) == {"auroc", "accuracy", "f1", "precision", "recall"}

    def test_perfect_predictions(self):
        """Perfect classification should give AUROC=1, accuracy=1."""
        y_true = [0, 0, 1, 1]
        y_pred = [0.0, 0.0, 1.0, 1.0]
        m = compute_metrics(y_true, y_pred, "classification")
        assert m["auroc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_single_class_auroc_nan(self):
        """AUROC should be NaN when y_true contains only one class."""
        m = compute_metrics([1, 1, 1], [0.5, 0.7, 0.9], "classification")
        assert np.isnan(m["auroc"])


class TestComputeMetricsEdgeCases:
    """Tests for compute_metrics error handling and edge cases."""

    def test_length_mismatch_raises(self):
        """Mismatched y_true / y_pred lengths should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_metrics([1, 2, 3], [1, 2], "regression")

    def test_empty_arrays_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_metrics([], [], "regression")

    def test_invalid_task_type_raises(self):
        """Invalid task_type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            compute_metrics([1, 2], [1, 2], "segmentation")


# ===================================================================
# Trainer (training/trainer.py)
# ===================================================================


class TestTrainerConstruction:
    """Tests for Trainer initialisation."""

    def test_default_params(self):
        """Default construction should work without errors."""
        model = _make_tiny_gnn()
        trainer = Trainer(model, task_type="regression")
        assert trainer.task_type == "regression"
        assert trainer.epochs == 100
        assert trainer.patience == 15

    def test_from_config(self):
        """from_config class method should build a Trainer from a dict."""
        model = _make_tiny_gnn()
        config = {"lr": 0.01, "epochs": 5, "patience": 2, "batch_size": 4}
        trainer = Trainer.from_config(model, config, task_type="regression")
        assert trainer.epochs == 5
        assert trainer.patience == 2
        assert trainer.batch_size == 4


class TestTrainerFitPredict:
    """Tests for Trainer fit/predict workflow on tiny data."""

    def test_fit_returns_history(self):
        """fit() should return a TrainHistory dataclass."""
        model = _make_tiny_gnn()
        trainer = Trainer(
            model, task_type="regression",
            lr=0.01, epochs=3, patience=10, batch_size=4,
        )
        train_graphs = _make_graph_dataset(_TINY_SMILES[:5], _TINY_TARGETS[:5])
        val_graphs = _make_graph_dataset(_TINY_SMILES[5:], _TINY_TARGETS[5:])

        history = trainer.fit(train_graphs, val_graphs)
        assert isinstance(history, TrainHistory)

    def test_history_populated(self):
        """TrainHistory should have non-empty train_loss, val_loss, val_metrics."""
        model = _make_tiny_gnn()
        trainer = Trainer(
            model, task_type="regression",
            lr=0.01, epochs=3, patience=10, batch_size=4,
        )
        train_graphs = _make_graph_dataset(_TINY_SMILES[:5], _TINY_TARGETS[:5])
        val_graphs = _make_graph_dataset(_TINY_SMILES[5:], _TINY_TARGETS[5:])

        history = trainer.fit(train_graphs, val_graphs)
        assert len(history.train_loss) == 3
        assert len(history.val_loss) == 3
        assert len(history.val_metrics) == 3
        assert all(isinstance(m, dict) for m in history.val_metrics)

    def test_best_epoch_set(self):
        """best_epoch should be set to a valid epoch index."""
        model = _make_tiny_gnn()
        trainer = Trainer(
            model, task_type="regression",
            lr=0.01, epochs=5, patience=10, batch_size=4,
        )
        train_graphs = _make_graph_dataset(_TINY_SMILES[:5], _TINY_TARGETS[:5])
        val_graphs = _make_graph_dataset(_TINY_SMILES[5:], _TINY_TARGETS[5:])

        history = trainer.fit(train_graphs, val_graphs)
        assert 0 <= history.best_epoch < 5

    def test_predict_length(self):
        """predict() should return one prediction per graph."""
        model = _make_tiny_gnn()
        trainer = Trainer(
            model, task_type="regression",
            lr=0.01, epochs=2, patience=10, batch_size=4,
        )
        train_graphs = _make_graph_dataset(_TINY_SMILES[:5], _TINY_TARGETS[:5])
        val_graphs = _make_graph_dataset(_TINY_SMILES[5:], _TINY_TARGETS[5:])
        trainer.fit(train_graphs, val_graphs)

        test_graphs = _make_graph_dataset(_TINY_SMILES[:3], _TINY_TARGETS[:3])
        preds = trainer.predict(test_graphs)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(test_graphs)


class TestTrainerEarlyStopping:
    """Tests for early stopping behaviour."""

    def test_early_stopping_triggers(self):
        """With patience=2 and enough epochs, training should stop early."""
        # Use a very high LR so training diverges quickly, triggering early stop
        model = _make_tiny_gnn()
        trainer = Trainer(
            model, task_type="regression",
            lr=0.5, epochs=50, patience=2, batch_size=4,
        )
        train_graphs = _make_graph_dataset(_TINY_SMILES[:5], _TINY_TARGETS[:5])
        val_graphs = _make_graph_dataset(_TINY_SMILES[5:], _TINY_TARGETS[5:])

        history = trainer.fit(train_graphs, val_graphs)
        # Should have stopped before reaching 50 epochs
        assert len(history.train_loss) < 50


# ===================================================================
# Evaluate (training/evaluate.py)
# ===================================================================


class TestEvaluateModelFP:
    """Tests for evaluate_model with FingerprintModel."""

    def test_returns_tuple(self):
        """evaluate_model should return (dict, DataFrame)."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal(20).astype(np.float32)

        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(X, y)

        metrics, pred_df = evaluate_model(
            model=model, test_data=X, y_true=y, task_type="regression",
        )
        assert isinstance(metrics, dict)
        assert isinstance(pred_df, pd.DataFrame)

    def test_regression_df_columns(self):
        """Regression predictions_df should have y_true, y_pred, error, abs_error."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 10)).astype(np.float32)
        y = rng.standard_normal(20).astype(np.float32)

        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(X, y)

        _, pred_df = evaluate_model(
            model=model, test_data=X, y_true=y, task_type="regression",
        )
        for col in ("y_true", "y_pred", "error", "abs_error"):
            assert col in pred_df.columns

    def test_smiles_column_when_provided(self):
        """predictions_df should have smiles column when smiles are given."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 10)).astype(np.float32)
        y = rng.standard_normal(5).astype(np.float32)
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)O", "CCCO"]

        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(X, y)

        _, pred_df = evaluate_model(
            model=model, test_data=X, y_true=y, task_type="regression",
            smiles=smiles,
        )
        assert "smiles" in pred_df.columns
        assert list(pred_df["smiles"]) == smiles


class TestEvaluateModelGNN:
    """Tests for evaluate_model with Trainer + GNN graphs."""

    def test_returns_tuple(self):
        """evaluate_model with Trainer should return (dict, DataFrame)."""
        model = _make_tiny_gnn()
        trainer = Trainer(
            model, task_type="regression",
            lr=0.01, epochs=2, patience=10, batch_size=4,
        )
        train_graphs = _make_graph_dataset(_TINY_SMILES[:5], _TINY_TARGETS[:5])
        val_graphs = _make_graph_dataset(_TINY_SMILES[5:], _TINY_TARGETS[5:])
        trainer.fit(train_graphs, val_graphs)

        test_graphs = _make_graph_dataset(_TINY_SMILES[:3], _TINY_TARGETS[:3])
        y_true = np.array(_TINY_TARGETS[:3])

        metrics, pred_df = evaluate_model(
            model=trainer, test_data=test_graphs, y_true=y_true,
            task_type="regression",
        )
        assert isinstance(metrics, dict)
        assert isinstance(pred_df, pd.DataFrame)
        assert len(pred_df) == 3


class TestEvaluateModelEdgeCases:
    """Tests for evaluate_model error handling."""

    def test_length_mismatch_raises(self):
        """Mismatched y_true and predictions should raise ValueError."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 5)).astype(np.float32)
        y_train = rng.standard_normal(10).astype(np.float32)
        y_wrong_len = np.array([1.0, 2.0, 3.0])  # only 3 vs 10 samples

        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(X, y_train)

        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_model(
                model=model, test_data=X, y_true=y_wrong_len,
                task_type="regression",
            )


class TestEvaluateTDC:
    """Tests for TDC evaluation functions (require data download)."""

    @pytest.mark.slow
    def test_evaluate_tdc_runs(self):
        """evaluate_tdc should run end-to-end (downloads data)."""
        # Intentionally not implemented inline — marked slow
        pytest.skip("TDC evaluation requires data download")

    @pytest.mark.slow
    def test_evaluate_tdc_multi_seed_runs(self):
        """evaluate_tdc_multi_seed should run end-to-end (downloads data)."""
        pytest.skip("TDC multi-seed evaluation requires data download")
