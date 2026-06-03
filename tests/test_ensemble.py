"""Tests for the ensemble module.

All tests use synthetic data — no TDC downloads, no network access.
GNN tests use tiny graphs (5 atoms, 3 molecules) to keep them fast.
Tests > 5 seconds are marked @pytest.mark.slow.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def regression_data(rng):
    """Small regression dataset: 80 train, 20 val, 20 test."""
    X_fp   = rng.random((120, 64)).astype(np.float32)   # mock fingerprints
    X_desc = rng.random((120, 8)).astype(np.float32)    # mock descriptors
    y      = (rng.random(120) * 4 - 2).astype(np.float64)  # [-2, 2]

    idx = np.arange(120)
    return {
        "X_fp_train": X_fp[:80],    "X_desc_train": X_desc[:80],    "y_train": y[:80],
        "X_fp_val":   X_fp[80:100], "X_desc_val":   X_desc[80:100], "y_val":   y[80:100],
        "X_fp_test":  X_fp[100:],   "X_desc_test":  X_desc[100:],   "y_test":  y[100:],
    }


@pytest.fixture
def classification_data(rng):
    """Small binary classification dataset."""
    X_fp   = rng.random((120, 64)).astype(np.float32)
    X_desc = rng.random((120, 8)).astype(np.float32)
    y      = rng.integers(0, 2, size=120).astype(np.float64)

    return {
        "X_fp_train": X_fp[:80],    "X_desc_train": X_desc[:80],    "y_train": y[:80],
        "X_fp_val":   X_fp[80:100], "X_desc_val":   X_desc[80:100], "y_val":   y[80:100],
        "X_fp_test":  X_fp[100:],   "X_desc_test":  X_desc[100:],   "y_test":  y[100:],
    }


def _make_fitted_models(X_fp_train, X_desc_train, y_train, task_type="regression"):
    """Create and fit two simple sklearn models for tests."""
    from molgate.models.sklearn_models import SklearnModel

    m_fp = SklearnModel(Ridge(), task_type=task_type, feature_type="morgan")
    m_desc = SklearnModel(Ridge(), task_type=task_type, feature_type="descriptors")
    if task_type == "regression":
        m_fp.fit(X_fp_train, y_train)
        m_desc.fit(X_desc_train, y_train)
    return {"model_fp": m_fp, "model_desc": m_desc}


def _make_unfitted_models(task_type="regression"):
    """Return two unfitted SklearnModel instances for ensemble tests."""
    from molgate.models.sklearn_models import SklearnModel
    return {
        "model_fp":   SklearnModel(Ridge(), task_type=task_type, feature_type="morgan"),
        "model_desc": SklearnModel(Ridge(), task_type=task_type, feature_type="descriptors"),
    }


def _x_dict(d, split="train"):
    """Build X_dict for a given split from regression/classification_data fixture."""
    return {
        "model_fp":   d[f"X_fp_{split}"],
        "model_desc": d[f"X_desc_{split}"],
    }


# ---------------------------------------------------------------------------
# Synthetic graph helper
# ---------------------------------------------------------------------------

def _make_synthetic_graphs(n: int = 6, n_atoms: int = 5, seed: int = 0):
    """Create n tiny PyG Data graphs for GNN tests."""
    import torch
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(n):
        x = torch.tensor(rng.random((n_atoms, 14)), dtype=torch.float)
        # Simple chain: 0-1-2-3-4 bidirectional
        src = [0, 1, 1, 2, 2, 3, 3, 4]
        dst = [1, 0, 2, 1, 3, 2, 4, 3]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(rng.random((len(src), 6)), dtype=torch.float)
        y = torch.tensor([rng.random()], dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs


# ---------------------------------------------------------------------------
# TestSklearnModel
# ---------------------------------------------------------------------------

class TestSklearnModel:
    def test_fit_predict_regression(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        m = SklearnModel(Ridge(), task_type="regression", feature_type="morgan")
        m.fit(d["X_fp_train"], d["y_train"])
        preds = m.predict(d["X_fp_test"])
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_feature_type_metadata(self):
        from molgate.models.sklearn_models import SklearnModel

        m = SklearnModel(Ridge(), task_type="regression", feature_type="descriptors",
                         fp_radius=3, fp_nbits=1024)
        assert m.feature_type == "descriptors"
        assert m.fp_radius == 3
        assert m.fp_nbits == 1024

    def test_get_params_includes_task_type(self):
        from molgate.models.sklearn_models import SklearnModel

        m = SklearnModel(Ridge(alpha=2.0), task_type="regression", feature_type="morgan")
        params = m.get_params()
        assert params["task_type"] == "regression"
        assert params["estimator_class"] == "Ridge"
        assert "alpha" in params

    def test_predict_proba_raises_for_regression(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        m = SklearnModel(Ridge(), task_type="regression", feature_type="morgan")
        m.fit(d["X_fp_train"], d["y_train"])
        with pytest.raises(RuntimeError, match="classification"):
            m.predict_proba(d["X_fp_test"])

    def test_random_forest(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        m = SklearnModel(
            RandomForestRegressor(n_estimators=10, random_state=0),
            task_type="regression",
            feature_type="morgan",
        )
        m.fit(d["X_fp_train"], d["y_train"])
        preds = m.predict(d["X_fp_test"])
        assert preds.shape == (20,)

    def test_repr(self):
        from molgate.models.sklearn_models import SklearnModel

        m = SklearnModel(Ridge(), task_type="regression", feature_type="descriptors")
        assert "Ridge" in repr(m)
        assert "regression" in repr(m)

    def test_tune_ridge_returns_best_params(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        m = SklearnModel(Ridge(), task_type="regression", feature_type="descriptors")
        best = m.tune(d["X_desc_train"], d["y_train"], n_trials=5, cv_folds=3, seed=0)
        assert "alpha" in best
        assert best["alpha"] > 0
        # Model should be fitted after tuning
        preds = m.predict(d["X_desc_test"])
        assert preds.shape == (20,)

    def test_tune_rf_returns_best_params(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        m = SklearnModel(
            RandomForestRegressor(n_estimators=10, random_state=0),
            task_type="regression",
            feature_type="morgan",
        )
        best = m.tune(d["X_fp_train"], d["y_train"], n_trials=3, cv_folds=3, seed=0)
        assert "n_estimators" in best
        preds = m.predict(d["X_fp_test"])
        assert preds.shape == (20,)

    def test_tune_unknown_estimator_raises(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel
        from sklearn.dummy import DummyRegressor

        d = regression_data
        m = SklearnModel(DummyRegressor(), task_type="regression", feature_type="morgan")
        with pytest.raises(ValueError, match="No search space defined"):
            m.tune(d["X_fp_train"], d["y_train"], n_trials=2, cv_folds=2)

    def test_get_sklearn_class_invalid(self):
        from molgate.models.sklearn_models import get_sklearn_class

        with pytest.raises(ValueError, match="Unknown sklearn class"):
            get_sklearn_class("NonExistentEstimator")

    def test_get_sklearn_class_valid(self):
        from molgate.models.sklearn_models import get_sklearn_class
        from sklearn.linear_model import Ridge as SklearnRidge

        cls = get_sklearn_class("Ridge")
        assert cls is SklearnRidge


# ---------------------------------------------------------------------------
# TestPredictHelper
# ---------------------------------------------------------------------------

class TestPredictHelper:
    def test_numpy_regression(self, regression_data):
        from molgate.models.sklearn_models import SklearnModel
        from molgate.models.ensemble import _get_predictions

        d = regression_data
        m = SklearnModel(Ridge(), task_type="regression", feature_type="morgan")
        m.fit(d["X_fp_train"], d["y_train"])
        preds = _get_predictions(m, d["X_fp_test"], "regression")
        assert preds.ndim == 1
        assert preds.shape[0] == 20

    def test_graph_routing(self):
        from molgate.models.ensemble import _get_predictions, _is_graph_input

        # _is_graph_input should detect lists
        assert _is_graph_input([1, 2, 3]) is True
        assert _is_graph_input(np.zeros((10, 5))) is False

    def test_is_not_graph_for_array(self):
        from molgate.models.ensemble import _is_graph_input

        assert _is_graph_input(np.zeros((5, 10))) is False


# ---------------------------------------------------------------------------
# TestVotingEnsemble
# ---------------------------------------------------------------------------

class TestVotingEnsemble:
    def test_fit_predict_regression(self, regression_data):
        from molgate.models.ensemble import VotingEnsemble

        d = regression_data
        ensemble = VotingEnsemble(_make_unfitted_models(), task_type="regression")
        ensemble.fit(_x_dict(d, "train"), d["y_train"])
        preds = ensemble.predict(_x_dict(d, "test"))
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_weighted_voting(self, regression_data):
        from molgate.models.ensemble import VotingEnsemble

        d = regression_data
        # Weight model_fp=0, model_desc=1 → should match model_desc alone
        ensemble_w = VotingEnsemble(
            _make_unfitted_models(), task_type="regression", weights=[0.0, 1.0]
        )
        ensemble_w.fit(_x_dict(d, "train"), d["y_train"])
        preds_w = ensemble_w.predict(_x_dict(d, "test"))

        from molgate.models.sklearn_models import SklearnModel
        m_desc = SklearnModel(Ridge(), task_type="regression", feature_type="descriptors")
        m_desc.fit(d["X_desc_train"], d["y_train"])
        preds_single = m_desc.predict(d["X_desc_test"])

        np.testing.assert_allclose(preds_w, preds_single, rtol=1e-5)

    def test_missing_key_raises(self, regression_data):
        from molgate.models.ensemble import VotingEnsemble

        d = regression_data
        ensemble = VotingEnsemble(_make_unfitted_models(), task_type="regression")
        # Provide only one key
        with pytest.raises(KeyError, match="model_fp"):
            ensemble.fit({"model_desc": d["X_desc_train"]}, d["y_train"])

    def test_predict_before_fit_raises(self, regression_data):
        from molgate.models.ensemble import VotingEnsemble

        d = regression_data
        ensemble = VotingEnsemble(_make_unfitted_models(), task_type="regression")
        with pytest.raises(RuntimeError, match="fit"):
            ensemble.predict(_x_dict(d, "test"))

    def test_single_model(self, regression_data):
        from molgate.models.ensemble import VotingEnsemble
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        models = {"only": SklearnModel(Ridge(), task_type="regression", feature_type="morgan")}
        ensemble = VotingEnsemble(models, task_type="regression")
        ensemble.fit({"only": d["X_fp_train"]}, d["y_train"])
        preds = ensemble.predict({"only": d["X_fp_test"]})
        assert preds.shape == (20,)

    def test_repr(self):
        from molgate.models.ensemble import VotingEnsemble

        e = VotingEnsemble(_make_unfitted_models(), task_type="regression")
        assert "VotingEnsemble" in repr(e)
        assert "regression" in repr(e)


# ---------------------------------------------------------------------------
# TestBlendingEnsemble
# ---------------------------------------------------------------------------

class TestBlendingEnsemble:
    def test_fit_predict_regression(self, regression_data):
        from molgate.models.ensemble import BlendingEnsemble

        d = regression_data
        ensemble = BlendingEnsemble(_make_unfitted_models(), task_type="regression")
        ensemble.fit(
            _x_dict(d, "train"), d["y_train"],
            _x_dict(d, "val"),   d["y_val"],
        )
        preds = ensemble.predict(_x_dict(d, "test"))
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_blend_weights_populated(self, regression_data):
        from molgate.models.ensemble import BlendingEnsemble

        d = regression_data
        ensemble = BlendingEnsemble(_make_unfitted_models(), task_type="regression")
        ensemble.fit(
            _x_dict(d, "train"), d["y_train"],
            _x_dict(d, "val"),   d["y_val"],
        )
        weights = ensemble.get_blend_weights()
        assert set(weights.keys()) == {"model_fp", "model_desc"}
        assert all(np.isfinite(v) for v in weights.values())

    def test_meta_model_is_ridge_for_regression(self, regression_data):
        from molgate.models.ensemble import BlendingEnsemble
        from sklearn.linear_model import Ridge as SKRidge

        d = regression_data
        ensemble = BlendingEnsemble(_make_unfitted_models(), task_type="regression",
                                    meta_alpha=2.0)
        ensemble.fit(
            _x_dict(d, "train"), d["y_train"],
            _x_dict(d, "val"),   d["y_val"],
        )
        assert isinstance(ensemble.meta_model, SKRidge)
        assert ensemble.meta_model.alpha == pytest.approx(2.0)

    def test_predict_before_fit_raises(self, regression_data):
        from molgate.models.ensemble import BlendingEnsemble

        d = regression_data
        ensemble = BlendingEnsemble(_make_unfitted_models(), task_type="regression")
        with pytest.raises(RuntimeError, match="fit"):
            ensemble.predict(_x_dict(d, "test"))

    def test_get_blend_weights_before_fit(self):
        from molgate.models.ensemble import BlendingEnsemble

        ensemble = BlendingEnsemble(_make_unfitted_models(), task_type="regression")
        assert ensemble.get_blend_weights() == {}

    def test_repr(self):
        from molgate.models.ensemble import BlendingEnsemble

        e = BlendingEnsemble(_make_unfitted_models(), task_type="regression", meta_alpha=5.0)
        assert "BlendingEnsemble" in repr(e)
        assert "5.0" in repr(e)


# ---------------------------------------------------------------------------
# TestStackingEnsemble
# ---------------------------------------------------------------------------

class TestStackingEnsemble:
    def test_fit_predict_regression(self, regression_data):
        from molgate.models.ensemble import StackingEnsemble

        d = regression_data
        ensemble = StackingEnsemble(
            _make_unfitted_models(), task_type="regression", n_folds=3
        )
        ensemble.fit(_x_dict(d, "train"), d["y_train"])
        preds = ensemble.predict(_x_dict(d, "test"))
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_meta_importances_populated(self, regression_data):
        from molgate.models.ensemble import StackingEnsemble

        d = regression_data
        ensemble = StackingEnsemble(
            _make_unfitted_models(), task_type="regression", n_folds=3
        )
        ensemble.fit(_x_dict(d, "train"), d["y_train"])
        importances = ensemble.get_meta_importances()
        assert set(importances.keys()) == {"model_fp", "model_desc"}
        assert all(np.isfinite(v) for v in importances.values())

    def test_refits_base_models_on_full_data(self, regression_data):
        """After fit(), predictions should come from fully-trained base models."""
        from molgate.models.ensemble import StackingEnsemble
        from molgate.models.sklearn_models import SklearnModel

        d = regression_data
        models = _make_unfitted_models()
        ensemble = StackingEnsemble(models, task_type="regression", n_folds=3)
        ensemble.fit(_x_dict(d, "train"), d["y_train"])

        # Base models should be fitted (predict without error)
        for name, model in ensemble.models.items():
            preds = model.predict(d[f"X_fp_test" if name == "model_fp" else "X_desc_test"])
            assert preds.shape == (20,)

    def test_predict_before_fit_raises(self, regression_data):
        from molgate.models.ensemble import StackingEnsemble

        d = regression_data
        ensemble = StackingEnsemble(_make_unfitted_models(), task_type="regression")
        with pytest.raises(RuntimeError, match="fit"):
            ensemble.predict(_x_dict(d, "test"))

    def test_gnn_hybrid_path(self):
        """GNNModelWrapper models should use the single-pass hybrid path in stacking."""
        from molgate.models.ensemble import GNNModelWrapper, StackingEnsemble
        from molgate.models.sklearn_models import SklearnModel

        graphs = _make_synthetic_graphs(n=30, n_atoms=5)

        rng = np.random.default_rng(0)
        X_desc = rng.random((30, 8)).astype(np.float32)
        y = np.array([float(g.y.item()) for g in graphs])

        class _FakeFastModel:
            """Minimal fit/predict for testing hybrid path without real GNN."""
            requires_graphs = False
            feature_type = "descriptors"

            def __init__(self):
                self._w = None

            def fit(self, X, y):
                self._w = np.linalg.lstsq(X, y, rcond=None)[0]
                return self

            def predict(self, X):
                return X @ self._w

        class _FakeGNNWrapper:
            """Simulates GNNModelWrapper (requires_graphs=True) without real GNN."""
            requires_graphs = True

            def fit(self, graphs, y=None):
                self._mean = np.mean([g.y.item() for g in graphs])
                return self

            def predict(self, graphs):
                return np.full(len(graphs), self._mean)

        models = {
            "fast": _FakeFastModel(),
            "gnn":  _FakeGNNWrapper(),
        }
        X_dict = {"fast": X_desc, "gnn": graphs}

        ensemble = StackingEnsemble(models, task_type="regression", n_folds=3)
        ensemble.fit(X_dict, y)

        # meta_feature_names should include both
        assert "fast" in ensemble._meta_feature_names
        assert "gnn" in ensemble._meta_feature_names

        preds = ensemble.predict(X_dict)
        assert preds.shape == (30,)
        assert np.isfinite(preds).all()

    def test_repr(self):
        from molgate.models.ensemble import StackingEnsemble

        e = StackingEnsemble(_make_unfitted_models(), task_type="regression",
                              n_folds=7, meta_alpha=0.5)
        assert "StackingEnsemble" in repr(e)
        assert "7" in repr(e)


# ---------------------------------------------------------------------------
# TestGNNModelWrapper
# ---------------------------------------------------------------------------

class TestGNNModelWrapper:
    @pytest.mark.slow
    def test_fit_predict(self):
        """Full GNNModelWrapper fit/predict with real MoleculeGNN."""
        from molgate.models.gnn import MoleculeGNN
        from molgate.models.ensemble import GNNModelWrapper

        graphs = _make_synthetic_graphs(n=20, n_atoms=5)
        training_config = {
            "lr": 0.01,
            "weight_decay": 0.0,
            "epochs": 3,
            "patience": 2,
            "batch_size": 8,
        }
        gnn = MoleculeGNN(hidden_dim=32, num_layers=2, task_type="regression")
        wrapper = GNNModelWrapper(gnn, training_config, task_type="regression",
                                  val_fraction=0.2, seed=0)
        wrapper.fit(graphs)
        preds = wrapper.predict(graphs)
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_requires_graphs_flag(self):
        from molgate.models.ensemble import GNNModelWrapper

        assert GNNModelWrapper.requires_graphs is True

    def test_predict_before_fit_raises(self):
        from molgate.models.gnn import MoleculeGNN
        from molgate.models.ensemble import GNNModelWrapper

        gnn = MoleculeGNN(hidden_dim=32, num_layers=2, task_type="regression")
        wrapper = GNNModelWrapper(gnn, {}, task_type="regression")
        with pytest.raises(RuntimeError, match="fit"):
            wrapper.predict([])

    def test_repr(self):
        from molgate.models.gnn import MoleculeGNN
        from molgate.models.ensemble import GNNModelWrapper

        gnn = MoleculeGNN(hidden_dim=32, num_layers=2, task_type="regression")
        wrapper = GNNModelWrapper(gnn, {}, task_type="regression")
        assert "GNNModelWrapper" in repr(wrapper)
        assert "False" in repr(wrapper)  # fitted=False


# ---------------------------------------------------------------------------
# TestFactoryEnsemble
# ---------------------------------------------------------------------------

class TestFactoryEnsemble:
    def test_create_sklearn_model(self):
        from molgate.models.factory import create_model
        from molgate.models.sklearn_models import SklearnModel

        m = create_model("rf_morgan", task_type="regression")
        assert isinstance(m, SklearnModel)
        assert m.feature_type == "morgan"

    def test_create_ridge_descriptors(self):
        from molgate.models.factory import create_model
        from molgate.models.sklearn_models import SklearnModel

        m = create_model("ridge_descriptors", task_type="regression")
        assert isinstance(m, SklearnModel)
        assert m.feature_type == "descriptors"

    def test_create_svr_descriptors(self):
        from molgate.models.factory import create_model
        from molgate.models.sklearn_models import SklearnModel

        m = create_model("svr_descriptors", task_type="regression")
        assert isinstance(m, SklearnModel)

    def test_create_voting_ensemble_fp(self):
        from molgate.models.factory import create_model
        from molgate.models.ensemble import VotingEnsemble

        e = create_model("ensemble_voting_fp", task_type="regression")
        assert isinstance(e, VotingEnsemble)
        # Should have 4 base models
        assert len(e.models) == 4
        assert "lgbm_morgan" in e.models
        assert "ridge_descriptors" in e.models

    def test_create_voting_ensemble_all_has_gnn_wrapper(self):
        from molgate.models.factory import create_model
        from molgate.models.ensemble import GNNModelWrapper, VotingEnsemble

        e = create_model("ensemble_voting_all", task_type="regression")
        assert isinstance(e, VotingEnsemble)
        assert "gnn" in e.models
        assert isinstance(e.models["gnn"], GNNModelWrapper)

    def test_create_blending_ensemble(self):
        from molgate.models.factory import create_model
        from molgate.models.ensemble import BlendingEnsemble

        e = create_model("ensemble_blending", task_type="regression")
        assert isinstance(e, BlendingEnsemble)
        assert len(e.models) == 3

    def test_create_stacking_ensemble(self):
        from molgate.models.factory import create_model
        from molgate.models.ensemble import StackingEnsemble

        e = create_model("ensemble_stacking", task_type="regression")
        assert isinstance(e, StackingEnsemble)
        assert e.n_folds == 5
        assert e.meta_alpha == pytest.approx(1.0)

    def test_unknown_model_raises(self):
        from molgate.models.factory import create_model

        with pytest.raises(KeyError, match="Unknown model name"):
            create_model("nonexistent_model")

    def test_list_models_includes_new_entries(self):
        from molgate.models.factory import list_models

        names = list_models()
        assert "rf_morgan" in names
        assert "ridge_descriptors" in names
        assert "svr_descriptors" in names
        assert "ensemble_voting_fp" in names
        assert "ensemble_stacking" in names


# ---------------------------------------------------------------------------
# Integration: end-to-end fit/predict with factory-created ensembles
# ---------------------------------------------------------------------------

class TestEnsembleIntegration:
    def test_voting_ensemble_fp_end_to_end(self, regression_data):
        from molgate.models.factory import create_model
        from molgate.models.ensemble import VotingEnsemble
        from molgate.data.featurizer import compute_fingerprints, compute_descriptors

        d = regression_data
        e = create_model("ensemble_voting_fp", task_type="regression")
        assert isinstance(e, VotingEnsemble)

        # Build X_dict matching the 4 base models (all numpy-array based)
        X_dict_train = {
            "lgbm_morgan":       d["X_fp_train"],
            "lgbm_descriptors":  d["X_desc_train"],
            "rf_morgan":         d["X_fp_train"],
            "ridge_descriptors": d["X_desc_train"],
        }
        X_dict_test = {
            "lgbm_morgan":       d["X_fp_test"],
            "lgbm_descriptors":  d["X_desc_test"],
            "rf_morgan":         d["X_fp_test"],
            "ridge_descriptors": d["X_desc_test"],
        }
        e.fit(X_dict_train, d["y_train"])
        preds = e.predict(X_dict_test)
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()

    def test_stacking_ensemble_end_to_end(self, regression_data):
        from molgate.models.factory import create_model
        from molgate.models.ensemble import StackingEnsemble

        d = regression_data
        e = create_model("ensemble_stacking", task_type="regression")
        assert isinstance(e, StackingEnsemble)

        X_dict_train = {
            "lgbm_morgan":       d["X_fp_train"],
            "lgbm_descriptors":  d["X_desc_train"],
            "rf_morgan":         d["X_fp_train"],
            "ridge_descriptors": d["X_desc_train"],
        }
        X_dict_test = {
            "lgbm_morgan":       d["X_fp_test"],
            "lgbm_descriptors":  d["X_desc_test"],
            "rf_morgan":         d["X_fp_test"],
            "ridge_descriptors": d["X_desc_test"],
        }
        e.fit(X_dict_train, d["y_train"])
        preds = e.predict(X_dict_test)
        assert preds.shape == (20,)
        assert np.isfinite(preds).all()
        assert e.get_meta_importances()  # non-empty
