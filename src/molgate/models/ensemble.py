"""Ensemble models for molecular property prediction.

Three complementary strategies ordered by complexity and data efficiency:

1. **VotingEnsemble** — unweighted (or weighted) average of base model predictions.
   Best for: quick baselines and when base models are similarly calibrated.

2. **BlendingEnsemble** — base models train on the training set; a Ridge / Logistic
   meta-model learns blend weights from their predictions on a held-out val set.
   Best for: when a natural train/val split already exists.

3. **StackingEnsemble** — K-fold out-of-fold (OOF) predictions from fast base models
   feed a meta-learner.  GNN models are trained once and added as extra columns
   rather than being re-trained per fold (too expensive).
   Best for: squeezing maximum performance from a fixed dataset.

All classes share a common interface::

    ensemble.fit(X_dict, y)                          # Voting / Stacking
    ensemble.fit(X_dict_train, y_train,
                 X_dict_val, y_val)                  # Blending
    preds = ensemble.predict(X_dict)                 # → 1-D np.ndarray

``X_dict`` maps each model name to its feature data::

    {
        "lgbm_morgan":      X_fingerprints,   # (N, 2048) np.ndarray
        "lgbm_descriptors": X_descriptors,    # (N, 12)   np.ndarray
        "rf_morgan":        X_fingerprints,   # same array, different model
        "gnn":              graphs,           # list[PyG Data] for GNNModelWrapper
    }

GNN support is provided via **GNNModelWrapper**, which bridges the Trainer's
graph-list interface into the unified fit/predict pattern.  The wrapper handles
its own internal train/val split so the ensemble API stays uniform.

For StackingEnsemble, GNN models participate via a hybrid strategy: they train
on the full training set (once) and their predictions are appended as fixed
columns to the OOF feature matrix.  This avoids K × full_GNN training cost
while still incorporating graph-level signal in the meta-learner.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GNNModelWrapper
# ---------------------------------------------------------------------------

class GNNModelWrapper:
    """Bridge between Trainer's graph-list interface and the ensemble's X_dict API.

    MoleculeGNN is a raw ``nn.Module``; training goes through ``Trainer``.
    This wrapper gives it the standard ``fit(graphs, y) / predict(graphs)``
    interface that VotingEnsemble, BlendingEnsemble, and StackingEnsemble expect.

    GNN early stopping requires a validation set.  Rather than changing the
    ensemble interface, the wrapper carves out an internal val split from the
    training graphs at fit time.

    ``requires_graphs = True`` is checked by StackingEnsemble to decide whether
    to use K-fold OOF (False) or the single-train hybrid path (True).

    Parameters
    ----------
    gnn_model : MoleculeGNN
        Unfitted PyG GNN model (created by factory.py).
    training_config : dict
        Training hyperparameters (lr, epochs, patience, batch_size, …),
        as read from models.yaml under the ``training`` key.
    task_type : str
        "regression" or "classification".
    val_fraction : float
        Fraction of training graphs reserved for internal validation / early stopping.
    seed : int
        RNG seed for the internal val split.
    """

    requires_graphs = True  # signals StackingEnsemble to use the hybrid path

    def __init__(
        self,
        gnn_model: Any,
        training_config: dict,
        task_type: str = "regression",
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        self.gnn_model = gnn_model
        self.training_config = training_config
        self.task_type = task_type
        self.val_fraction = val_fraction
        self.seed = seed
        self.trainer: Any = None
        self._fitted = False

    def fit(self, graphs: list, y: Any = None) -> "GNNModelWrapper":
        """Train the GNN with an internal train/val split for early stopping.

        ``y`` is ignored — targets are embedded in each graph's ``.y`` attribute
        by the featurizer.  It is accepted only to keep the interface uniform.

        Parameters
        ----------
        graphs : list[PyG Data]
            Training graphs with ``.y`` attributes set.
        y : ignored
            Present for API compatibility only.
        """
        from molgate.training.trainer import Trainer

        n = len(graphs)
        n_val = max(1, int(n * self.val_fraction))
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        val_graphs   = [graphs[i] for i in idx[:n_val]]
        train_graphs = [graphs[i] for i in idx[n_val:]]

        logger.info(
            f"GNNModelWrapper: training on {len(train_graphs)} graphs, "
            f"validating on {len(val_graphs)} graphs"
        )

        self.trainer = Trainer.from_config(
            self.gnn_model, self.training_config, task_type=self.task_type
        )
        self.trainer.fit(train_graphs, val_graphs)
        self._fitted = True
        return self

    def predict(self, graphs: list) -> np.ndarray:
        """Return predictions as a 1-D numpy array."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        return self.trainer.predict(graphs)

    def predict_proba(self, graphs: list) -> np.ndarray:
        """Return class probabilities (classification only).

        GNN forward() already applies sigmoid, so the output is in [0, 1]
        and can be used directly as a positive-class probability.
        """
        return self.predict(graphs)

    def __repr__(self) -> str:
        return (
            f"GNNModelWrapper(task_type={self.task_type!r}, "
            f"val_fraction={self.val_fraction}, fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_graph_input(X: Any) -> bool:
    """Return True if X is a list of PyG Data objects (graph input)."""
    return isinstance(X, list)


def _get_predictions(model: Any, X: Any, task_type: str) -> np.ndarray:
    """Dispatch to the right predict method and return a 1-D array.

    Routing:
    - Graph list → GNNModelWrapper path (predict / predict_proba)
    - numpy array + classification + predict_proba exists → predict_proba[:, 1]
    - numpy array + regression → predict
    """
    if _is_graph_input(X):
        if task_type == "classification":
            return np.asarray(model.predict_proba(X)).ravel()
        return np.asarray(model.predict(X)).ravel()

    # numpy-array path (FingerprintModel, SklearnModel, …)
    if task_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()
    return np.asarray(model.predict(X)).ravel()


def _fit_model(model: Any, X: Any, y: np.ndarray) -> None:
    """Call model.fit with the right signature based on input type."""
    if _is_graph_input(X):
        model.fit(X)          # GNNModelWrapper: y embedded in graphs
    else:
        model.fit(X, y)       # FingerprintModel / SklearnModel


# ---------------------------------------------------------------------------
# VotingEnsemble
# ---------------------------------------------------------------------------

class VotingEnsemble:
    """Average predictions from multiple base models.

    The simplest possible ensemble: each base model votes with its prediction
    and the final output is a (optionally weighted) mean.  Works well when:

    - Base models have low prediction correlation — e.g., fingerprints capture
      local substructure while descriptors capture global physicochemical context.
    - Models are similarly calibrated (all on the same output scale).

    Supports mixing numpy-array models (FingerprintModel, SklearnModel) and
    graph-list models (GNNModelWrapper) in the same ensemble via ``X_dict``.

    For regression: weighted mean of predictions.
    For classification: weighted mean of predicted probabilities, thresholded at 0.5.

    Parameters
    ----------
    models : dict[str, Any]
        Unfitted model instances keyed by name.
    task_type : str
        "regression" or "classification".
    weights : list[float], optional
        Per-model weights (same order as ``models``). Defaults to equal weights.
    """

    def __init__(
        self,
        models: dict[str, Any],
        task_type: str = "regression",
        weights: list[float] | None = None,
    ):
        self.models = models
        self.task_type = task_type
        self.weights = weights
        self._fitted = False

    def fit(self, X_dict: dict[str, Any], y: np.ndarray) -> "VotingEnsemble":
        """Train all base models.

        Parameters
        ----------
        X_dict : dict[str, Any]
            Feature data keyed by model name.  Values are either numpy arrays
            (FingerprintModel / SklearnModel) or graph lists (GNNModelWrapper).
        y : np.ndarray
            Target values. Ignored for GNN models (targets live in graphs).
        """
        for name, model in self.models.items():
            if name not in X_dict:
                raise KeyError(
                    f"Model '{name}' missing from X_dict. "
                    f"Available keys: {list(X_dict.keys())}"
                )
            logger.info(f"  Fitting base model: {name}")
            _fit_model(model, X_dict[name], y)

        self._fitted = True
        logger.info(f"VotingEnsemble fitted ({len(self.models)} base models)")
        return self

    def predict(self, X_dict: dict[str, Any]) -> np.ndarray:
        """Return ensemble predictions (weighted mean, then threshold for classification)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        all_preds = [
            _get_predictions(model, X_dict[name], self.task_type)
            for name, model in self.models.items()
        ]
        avg = (
            np.average(all_preds, axis=0, weights=self.weights)
            if self.weights is not None
            else np.mean(all_preds, axis=0)
        )
        return (avg >= 0.5).astype(int) if self.task_type == "classification" else avg

    def predict_proba(self, X_dict: dict[str, Any]) -> np.ndarray:
        """Return averaged predicted probabilities (classification only)."""
        all_preds = [
            _get_predictions(model, X_dict[name], "classification")
            for name, model in self.models.items()
        ]
        return (
            np.average(all_preds, axis=0, weights=self.weights)
            if self.weights is not None
            else np.mean(all_preds, axis=0)
        )

    def __repr__(self) -> str:
        return (
            f"VotingEnsemble(models={list(self.models.keys())}, "
            f"task_type={self.task_type!r}, weights={self.weights})"
        )


# ---------------------------------------------------------------------------
# BlendingEnsemble
# ---------------------------------------------------------------------------

class BlendingEnsemble:
    """Blend base model predictions using a learned meta-model.

    Unlike voting, blending learns optimal combination weights from a held-out
    validation set.  The meta-model (Ridge for regression, LogisticRegression
    for classification) maps stacked base model predictions to the final output.

    Training procedure:
    1. Fit all base models on ``X_dict_train`` / ``y_train``.
    2. Generate validation predictions from each base model → meta-features (N_val, n_models).
    3. Fit the meta-model: meta-features → ``y_val``.

    For GNNModelWrapper base models: the wrapper trains with its internal val split;
    the external ``X_dict_val`` graphs are then used *only* to generate meta-features
    for the blending meta-model (not for GNN training).

    Parameters
    ----------
    models : dict[str, Any]
        Unfitted model instances keyed by name.
    task_type : str
        "regression" or "classification".
    meta_alpha : float
        Ridge regularization / inverse LogisticRegression C.
        Higher = blend weights pulled closer to equal.
    """

    def __init__(
        self,
        models: dict[str, Any],
        task_type: str = "regression",
        meta_alpha: float = 1.0,
    ):
        self.models = models
        self.task_type = task_type
        self.meta_alpha = meta_alpha
        self.meta_model: Any = None
        self._fitted = False

    def fit(
        self,
        X_dict_train: dict[str, Any],
        y_train: np.ndarray,
        X_dict_val: dict[str, Any],
        y_val: np.ndarray,
    ) -> "BlendingEnsemble":
        """Fit base models on train; learn meta-model from val predictions.

        Parameters
        ----------
        X_dict_train : dict[str, Any]
            Training features for each model.
        y_train : np.ndarray
            Training targets.
        X_dict_val : dict[str, Any]
            Validation features — used to generate meta-learner training data.
        y_val : np.ndarray
            Validation targets.
        """
        for name, model in self.models.items():
            logger.info(f"  Fitting base model: {name}")
            _fit_model(model, X_dict_train[name], y_train)

        val_meta_X = self._make_meta_features(X_dict_val)
        logger.info(f"  Meta-features shape: {val_meta_X.shape}")

        if self.task_type == "classification":
            self.meta_model = LogisticRegression(
                C=1.0 / self.meta_alpha, max_iter=500, solver="lbfgs"
            )
        else:
            self.meta_model = Ridge(alpha=self.meta_alpha)

        self.meta_model.fit(val_meta_X, y_val)
        logger.info(f"  Blend weights: {self.get_blend_weights()}")

        self._fitted = True
        logger.info(f"BlendingEnsemble fitted ({len(self.models)} base models)")
        return self

    def predict(self, X_dict: dict[str, Any]) -> np.ndarray:
        """Return blended predictions via the meta-model."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        preds = self.meta_model.predict(self._make_meta_features(X_dict))
        return (preds >= 0.5).astype(int) if self.task_type == "classification" else preds

    def predict_proba(self, X_dict: dict[str, Any]) -> np.ndarray:
        """Return blended probabilities (classification only)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        meta_X = self._make_meta_features(X_dict)
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_X)[:, 1]
        return self.meta_model.predict(meta_X)

    def get_blend_weights(self) -> dict[str, float]:
        """Return the learned blend coefficient for each base model."""
        if self.meta_model is None:
            return {}
        coef = getattr(self.meta_model, "coef_", np.array([]))
        return dict(zip(self.models.keys(), coef.flat))

    def _make_meta_features(self, X_dict: dict[str, Any]) -> np.ndarray:
        cols = [
            _get_predictions(model, X_dict[name], self.task_type)
            for name, model in self.models.items()
        ]
        return np.column_stack(cols)

    def __repr__(self) -> str:
        return (
            f"BlendingEnsemble(models={list(self.models.keys())}, "
            f"task_type={self.task_type!r}, meta_alpha={self.meta_alpha})"
        )


# ---------------------------------------------------------------------------
# StackingEnsemble
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """K-fold stacking with a hybrid strategy for expensive models (GNN).

    Standard stacking (for fast models — FingerprintModel, SklearnModel):
    1. For each of K folds, deep-copy base models, train on (K-1) folds,
       predict on the held-out fold → out-of-fold (OOF) predictions.
    2. Stack OOF predictions → (N_train, n_fast_models) meta-feature matrix.
    3. Fit meta-learner on OOF features → training targets.
    4. Re-fit all fast base models on the full dataset for inference.

    Hybrid extension for graph models (GNNModelWrapper, ``requires_graphs=True``):
    - K-fold GNN training is prohibitive (K × 100+ epochs).
    - Instead: train each GNN model once on the full dataset, compute predictions
      on the training set, and append them as extra columns to the OOF matrix.
    - These columns are constant across folds (no cross-validation) but the
      meta-learner can still learn a useful coefficient for them.

    Parameters
    ----------
    models : dict[str, Any]
        Unfitted model instances keyed by name.  Deep-copied per fold for fast
        models; trained once for GNN models.
    task_type : str
        "regression" or "classification".
    n_folds : int
        Number of CV folds for OOF generation of fast models (default 5).
    meta_alpha : float
        Regularization for the Ridge / LogisticRegression meta-model.
    seed : int
        Random seed for KFold shuffle.
    """

    def __init__(
        self,
        models: dict[str, Any],
        task_type: str = "regression",
        n_folds: int = 5,
        meta_alpha: float = 1.0,
        seed: int = 42,
    ):
        self.models = models
        self.task_type = task_type
        self.n_folds = n_folds
        self.meta_alpha = meta_alpha
        self.seed = seed
        self.meta_model: Any = None
        self._fitted = False

    def fit(self, X_dict: dict[str, Any], y: np.ndarray) -> "StackingEnsemble":
        """Generate OOF predictions, fit meta-learner, refit all base models.

        Parameters
        ----------
        X_dict : dict[str, Any]
            Feature data for each model.
        y : np.ndarray
            Training targets.
        """
        n = len(y)
        fast_names = [
            name for name, m in self.models.items()
            if not getattr(m, "requires_graphs", False)
        ]
        graph_names = [
            name for name, m in self.models.items()
            if getattr(m, "requires_graphs", False)
        ]

        # --- OOF predictions for fast models (K-fold) ---
        oof_cols: dict[str, np.ndarray] = {name: np.zeros(n) for name in fast_names}
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        if fast_names:
            logger.info(
                f"Generating OOF predictions ({self.n_folds} folds × {len(fast_names)} fast models)..."
            )
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y)):
                for name in fast_names:
                    fold_model = copy.deepcopy(self.models[name])
                    fold_model.fit(X_dict[name][train_idx], y[train_idx])
                    oof_cols[name][val_idx] = _get_predictions(
                        fold_model, X_dict[name][val_idx], self.task_type
                    )
                logger.info(f"  Fold {fold_idx + 1}/{self.n_folds} complete")

        # --- Single-pass predictions for graph models (hybrid path) ---
        graph_cols: dict[str, np.ndarray] = {}
        if graph_names:
            logger.info(
                f"Training {len(graph_names)} graph model(s) once (hybrid stacking path)..."
            )
            for name in graph_names:
                _fit_model(self.models[name], X_dict[name], y)
                graph_cols[name] = _get_predictions(
                    self.models[name], X_dict[name], self.task_type
                )
                logger.info(f"  {name}: trained and predicted on full training set")

        # --- Build meta-feature matrix ---
        all_cols = (
            [oof_cols[n] for n in fast_names]
            + [graph_cols[n] for n in graph_names]
        )
        all_names = fast_names + graph_names
        oof_matrix = np.column_stack(all_cols)
        self._meta_feature_names = all_names
        logger.info(f"  Meta-feature matrix: {oof_matrix.shape} ({all_names})")

        # --- Fit meta-learner ---
        if self.task_type == "classification":
            self.meta_model = LogisticRegression(
                C=1.0 / self.meta_alpha, max_iter=500, solver="lbfgs"
            )
        else:
            self.meta_model = Ridge(alpha=self.meta_alpha)
        self.meta_model.fit(oof_matrix, y)
        logger.info(f"  Meta importances: {self.get_meta_importances()}")

        # --- Refit fast base models on full dataset for inference ---
        logger.info("Re-fitting fast base models on full training set...")
        for name in fast_names:
            self.models[name].fit(X_dict[name], y)
            logger.info(f"  Re-fitted: {name}")

        self._fitted = True
        logger.info("StackingEnsemble fitted")
        return self

    def predict(self, X_dict: dict[str, Any]) -> np.ndarray:
        """Return stacked predictions via the meta-learner."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        preds = self.meta_model.predict(self._make_meta_features(X_dict))
        return (preds >= 0.5).astype(int) if self.task_type == "classification" else preds

    def predict_proba(self, X_dict: dict[str, Any]) -> np.ndarray:
        """Return stacked probabilities (classification only)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        meta_X = self._make_meta_features(X_dict)
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_X)[:, 1]
        return self.meta_model.predict(meta_X)

    def get_meta_importances(self) -> dict[str, float]:
        """Return meta-learner coefficients as a proxy for base model importance."""
        if self.meta_model is None:
            return {}
        coef = getattr(self.meta_model, "coef_", np.array([]))
        names = getattr(self, "_meta_feature_names", list(self.models.keys()))
        return dict(zip(names, coef.flat))

    def _make_meta_features(self, X_dict: dict[str, Any]) -> np.ndarray:
        names = getattr(self, "_meta_feature_names", list(self.models.keys()))
        cols = [
            _get_predictions(self.models[n], X_dict[n], self.task_type)
            for n in names
        ]
        return np.column_stack(cols)

    def __repr__(self) -> str:
        return (
            f"StackingEnsemble(models={list(self.models.keys())}, "
            f"task_type={self.task_type!r}, n_folds={self.n_folds}, "
            f"meta_alpha={self.meta_alpha})"
        )


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.featurizer import compute_descriptors, compute_fingerprints
    from molgate.data.loaders import load_dataset
    from molgate.data.splits import random_split
    from molgate.models.factory import create_model
    from molgate.training.metrics import compute_metrics

    print("Loading solubility dataset...")
    df = load_dataset("solubility")
    splits = random_split(df, seed=42)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    def _features(df_):
        sm = df_["smiles"].tolist()
        return {
            "lgbm_morgan":      compute_fingerprints(sm),
            "lgbm_descriptors": compute_descriptors(sm).values,
            "rf_morgan":        compute_fingerprints(sm),
            "ridge_descriptors": compute_descriptors(sm).values,
        }

    X_train = _features(train_df)
    X_val   = _features(val_df)
    X_test  = _features(test_df)
    y_train = train_df["y"].values
    y_val   = val_df["y"].values
    y_test  = test_df["y"].values

    def make_models():
        return {
            "lgbm_morgan":      create_model("lgbm_morgan",      task_type="regression"),
            "lgbm_descriptors": create_model("lgbm_descriptors", task_type="regression"),
            "rf_morgan":        create_model("rf_morgan",         task_type="regression"),
            "ridge_descriptors":create_model("ridge_descriptors", task_type="regression"),
        }

    print("\n--- Individual baselines ---")
    for name, X_tr, X_te in [
        ("lgbm_morgan",      X_train["lgbm_morgan"],      X_test["lgbm_morgan"]),
        ("lgbm_descriptors", X_train["lgbm_descriptors"], X_test["lgbm_descriptors"]),
        ("rf_morgan",        X_train["rf_morgan"],        X_test["rf_morgan"]),
        ("ridge_descriptors",X_train["ridge_descriptors"],X_test["ridge_descriptors"]),
    ]:
        m = create_model(name, task_type="regression")
        m.fit(X_tr, y_train)
        met = compute_metrics(y_test, m.predict(X_te), task_type="regression")
        print(f"  {name:<20s}  MAE={met['mae']:.4f}  RMSE={met['rmse']:.4f}")

    print("\n--- VotingEnsemble ---")
    voting = VotingEnsemble(make_models(), task_type="regression")
    voting.fit(X_train, y_train)
    met = compute_metrics(y_test, voting.predict(X_test), task_type="regression")
    print(f"  MAE={met['mae']:.4f}  RMSE={met['rmse']:.4f}")

    print("\n--- BlendingEnsemble ---")
    blending = BlendingEnsemble(make_models(), task_type="regression", meta_alpha=1.0)
    blending.fit(X_train, y_train, X_val, y_val)
    print(f"  Blend weights: {blending.get_blend_weights()}")
    met = compute_metrics(y_test, blending.predict(X_test), task_type="regression")
    print(f"  MAE={met['mae']:.4f}  RMSE={met['rmse']:.4f}")

    print("\n--- StackingEnsemble ---")
    stacking = StackingEnsemble(make_models(), task_type="regression", n_folds=5)
    stacking.fit(X_train, y_train)
    print(f"  Meta importances: {stacking.get_meta_importances()}")
    met = compute_metrics(y_test, stacking.predict(X_test), task_type="regression")
    print(f"  MAE={met['mae']:.4f}  RMSE={met['rmse']:.4f}")

    import IPython
    IPython.embed()
