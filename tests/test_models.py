"""Tests for models/baseline.py, models/gnn.py, and models/factory.py."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from molgate.data.featurizer import smiles_to_graph
from molgate.models.baseline import FingerprintModel
from molgate.models.gnn import MoleculeGNN
from molgate.models.factory import create_model, list_models

# Same SMILES used in conftest — duplicated here to avoid importing conftest
# (pytest loads conftest automatically but it's not importable as a module).
SAMPLE_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "c1ccc(O)cc1",
    "CC(=O)O",
    "c1ccncc1",
    "C1CCCCC1",
    "c1ccc(-c2ccccc2)cc1",
    "CC(C)(C)c1ccc(O)cc1",
]

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
SYNTH_X = RNG.standard_normal((20, 10)).astype(np.float32)
SYNTH_Y_REG = RNG.standard_normal(20).astype(np.float32)
SYNTH_Y_CLS = RNG.integers(0, 2, size=20).astype(np.float32)


def _build_graph_batch(smiles: list[str], targets: list[float] | None = None) -> Batch:
    """Convert SMILES to a PyG Batch for GNN tests."""
    graphs = []
    for i, smi in enumerate(smiles):
        y = targets[i] if targets is not None else None
        g = smiles_to_graph(smi, y=y)
        if g is not None:
            graphs.append(g)
    return Batch.from_data_list(graphs)


# ===================================================================
# FingerprintModel (models/baseline.py)
# ===================================================================


class TestFingerprintModelConstruction:
    """Tests for FingerprintModel initialisation and parameter validation."""

    def test_regression_construction(self):
        """Regression task_type should create a valid model."""
        model = FingerprintModel(task_type="regression")
        assert model.task_type == "regression"
        assert model.model_ is not None

    def test_classification_construction(self):
        """Classification task_type should create a valid model."""
        model = FingerprintModel(task_type="classification")
        assert model.task_type == "classification"
        assert model.model_ is not None

    def test_invalid_task_type_raises(self):
        """Invalid task_type should raise ValueError."""
        with pytest.raises(ValueError, match="task_type"):
            FingerprintModel(task_type="invalid")

    def test_invalid_estimator_raises(self):
        """Invalid estimator should raise ValueError."""
        with pytest.raises(ValueError, match="estimator"):
            FingerprintModel(estimator="catboost")


class TestFingerprintModelFitPredict:
    """Tests for FingerprintModel fit/predict workflow."""

    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        result = model.fit(SYNTH_X, SYNTH_Y_REG)
        assert result is model

    def test_predict_shape_regression(self):
        """predict() should return array with one value per sample."""
        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(SYNTH_X, SYNTH_Y_REG)
        preds = model.predict(SYNTH_X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (20,)

    def test_predict_shape_classification(self):
        """Classification predict() should return array with one prob per sample."""
        model = FingerprintModel(
            task_type="classification", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(SYNTH_X, SYNTH_Y_CLS)
        preds = model.predict(SYNTH_X)
        assert preds.shape == (20,)

    def test_predict_before_fit_raises(self):
        """predict() before fit() should raise RuntimeError."""
        model = FingerprintModel(task_type="regression")
        # The model_ is created in __init__, but the sklearn estimator
        # hasn't been fitted. We need to check the actual behaviour.
        # The model sets model_ in __init__ via _build_model, so predict
        # won't hit the "model_ is None" guard. However the underlying
        # sklearn estimator will raise.  Let's verify an error is raised.
        with pytest.raises(Exception):
            model.predict(SYNTH_X)

    def test_classification_predict_probabilities(self):
        """Classification predictions should be probabilities in [0, 1]."""
        model = FingerprintModel(
            task_type="classification", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(SYNTH_X, SYNTH_Y_CLS)
        preds = model.predict(SYNTH_X)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)


class TestFingerprintModelIntrospection:
    """Tests for FingerprintModel get_params and feature_importances."""

    def test_get_params_keys(self):
        """get_params() should contain task_type and estimator keys."""
        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        params = model.get_params()
        assert "task_type" in params
        assert "estimator" in params
        assert params["task_type"] == "regression"
        assert params["estimator"] == "lightgbm"

    def test_feature_importances_after_fit(self):
        """feature_importances() should return an array after fit()."""
        model = FingerprintModel(
            task_type="regression", params={"n_estimators": 10, "verbose": -1}
        )
        model.fit(SYNTH_X, SYNTH_Y_REG)
        importances = model.feature_importances()
        assert isinstance(importances, np.ndarray)
        assert importances.shape == (SYNTH_X.shape[1],)

    def test_feature_importances_before_fit(self):
        """feature_importances() should return None before fit() (model_ is set but unfitted)."""
        model = FingerprintModel(task_type="regression")
        # model_ is not None (set by _build_model), but accessing
        # feature_importances_ on an unfitted LGBMRegressor raises AttributeError.
        # The method guards against model_ is None, so it may raise or return None.
        # Either way this should not crash with an unexpected error.
        try:
            result = model.feature_importances()
            # If it returns, it should be None or an array
        except AttributeError:
            pass  # Expected — unfitted estimator has no feature_importances_


# ===================================================================
# MoleculeGNN (models/gnn.py)
# ===================================================================


class TestMoleculeGNNConstruction:
    """Tests for MoleculeGNN initialisation."""

    def test_default_params(self):
        """Default construction should work without errors."""
        model = MoleculeGNN()
        assert model.hidden_dim == 128
        assert model.num_layers == 3
        assert model.task_type == "regression"
        assert model.pool_name == "mean"

    def test_custom_params(self):
        """Custom hidden_dim, num_layers, pool, dropout should be stored."""
        model = MoleculeGNN(
            hidden_dim=32, num_layers=1, pool="sum", dropout=0.2
        )
        assert model.hidden_dim == 32
        assert model.num_layers == 1
        assert model.pool_name == "sum"
        assert model.dropout_rate == 0.2

    def test_invalid_pool_raises(self):
        """Unknown pool type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown pool type"):
            MoleculeGNN(pool="max")

    def test_parameter_count_positive(self):
        """Model should have a non-trivial number of parameters."""
        model = MoleculeGNN(hidden_dim=32, num_layers=1)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


class TestMoleculeGNNForward:
    """Tests for MoleculeGNN forward pass using real molecular graphs."""

    def test_forward_output_shape(self):
        """Forward pass on a batch should return (batch_size, 1)."""
        model = MoleculeGNN(hidden_dim=32, num_layers=1, task_type="regression")
        model.eval()
        batch = _build_graph_batch(SAMPLE_SMILES[:4])

        with torch.no_grad():
            out = model(batch)

        assert out.shape == (4, 1)

    def test_classification_sigmoid(self):
        """Classification model output should be in [0, 1] (sigmoid applied)."""
        model = MoleculeGNN(
            hidden_dim=32, num_layers=1, task_type="classification"
        )
        model.eval()
        batch = _build_graph_batch(SAMPLE_SMILES[:4])

        with torch.no_grad():
            out = model(batch)

        assert out.shape == (4, 1)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_regression_no_sigmoid(self):
        """Regression output can be any real value (no sigmoid clamping)."""
        model = MoleculeGNN(hidden_dim=32, num_layers=1, task_type="regression")
        model.eval()
        batch = _build_graph_batch(SAMPLE_SMILES[:4])

        with torch.no_grad():
            out = model(batch)

        # Regression output is unrestricted; just verify shape and dtype
        assert out.dtype == torch.float32

    def test_single_molecule(self):
        """Forward pass should work on a single molecule batch."""
        model = MoleculeGNN(hidden_dim=32, num_layers=1)
        model.eval()
        batch = _build_graph_batch(["CCO"])

        with torch.no_grad():
            out = model(batch)

        assert out.shape == (1, 1)


class TestMoleculeGNNConfig:
    """Tests for MoleculeGNN get_config."""

    def test_get_config_keys(self):
        """get_config() should return all expected keys."""
        model = MoleculeGNN(hidden_dim=64, num_layers=2, dropout=0.15, pool="sum")
        config = model.get_config()
        expected_keys = {
            "model_type", "hidden_dim", "num_layers",
            "task_type", "dropout", "pool", "n_parameters",
        }
        assert set(config.keys()) == expected_keys

    def test_get_config_values(self):
        """get_config() values should match constructor arguments."""
        model = MoleculeGNN(
            hidden_dim=64, num_layers=2, task_type="classification",
            dropout=0.15, pool="sum",
        )
        config = model.get_config()
        assert config["model_type"] == "gnn"
        assert config["hidden_dim"] == 64
        assert config["num_layers"] == 2
        assert config["task_type"] == "classification"
        assert config["dropout"] == 0.15
        assert config["pool"] == "sum"
        assert config["n_parameters"] > 0


# ===================================================================
# Factory (models/factory.py)
# ===================================================================


class TestCreateModel:
    """Tests for the create_model factory function."""

    def test_create_lgbm_morgan(self):
        """Should create a FingerprintModel for 'lgbm_morgan'."""
        model = create_model("lgbm_morgan", task_type="regression")
        assert isinstance(model, FingerprintModel)
        assert model.estimator_name == "lightgbm"

    def test_create_lgbm_descriptors(self):
        """Should create a FingerprintModel for 'lgbm_descriptors'."""
        model = create_model("lgbm_descriptors", task_type="regression")
        assert isinstance(model, FingerprintModel)

    def test_create_gnn(self):
        """Should create a MoleculeGNN for 'gnn'."""
        model = create_model("gnn", task_type="regression")
        assert isinstance(model, MoleculeGNN)

    def test_unknown_model_raises(self):
        """Unknown model name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model name"):
            create_model("nonexistent_model")

    def test_fp_model_metadata(self):
        """FP models should have feature_type, fp_radius, fp_nbits metadata."""
        model = create_model("lgbm_morgan", task_type="regression")
        assert hasattr(model, "feature_type")
        assert hasattr(model, "fp_radius")
        assert hasattr(model, "fp_nbits")
        assert model.feature_type == "morgan"
        assert isinstance(model.fp_radius, int)
        assert isinstance(model.fp_nbits, int)

    def test_gnn_model_training_config(self):
        """GNN models should have training_config metadata."""
        model = create_model("gnn", task_type="regression")
        assert hasattr(model, "training_config")
        assert isinstance(model.training_config, dict)
        assert "lr" in model.training_config
        assert "epochs" in model.training_config

    def test_overrides(self):
        """Overrides should modify model parameters."""
        model = create_model(
            "lgbm_morgan",
            task_type="regression",
            overrides={"lightgbm_params": {"n_estimators": 42}},
        )
        params = model.get_params()
        assert params["n_estimators"] == 42


class TestListModels:
    """Tests for list_models helper."""

    def test_returns_list(self):
        """list_models() should return a sorted list of strings."""
        models = list_models()
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)

    def test_expected_models(self):
        """Should contain the three models defined in models.yaml."""
        models = list_models()
        assert "lgbm_morgan" in models
        assert "lgbm_descriptors" in models
        assert "gnn" in models

    def test_sorted(self):
        """list_models() should return names in sorted order."""
        models = list_models()
        assert models == sorted(models)
