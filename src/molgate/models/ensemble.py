"""Ensemble models for molecular property prediction.

Three complementary strategies ordered by complexity:

1. VotingEnsemble — unweighted (or weighted) average of base model predictions.
   Best for: quick baselines; when base models are similarly calibrated.

2. BlendingEnsemble — base models train on the training set; a Ridge meta-model
   learns optimal blend weights from their predictions on a held-out val set.
   Best for: when a natural train/val split exists and you want learned weights.

3. StackingEnsemble — K-fold out-of-fold (OOF) predictions feed a meta-learner
   trained on OOF targets.  More data-efficient than blending because every
   training sample contributes to the meta-learner via cross-validation.
   Best for: squeezing out the last error points when data volume is modest.

All classes share a common interface::

    ensemble.fit(X_dict, y)                      # Voting / Stacking
    ensemble.fit(X_dict_train, y_train,
                 X_dict_val, y_val)              # Blending
    preds = ensemble.predict(X_dict)             # → 1-D np.ndarray

Where ``X_dict`` maps each model name to its feature matrix::

    {
        "lgbm_morgan":      X_fingerprints,   # (N, 2048) numpy array
        "lgbm_descriptors": X_descriptors,    # (N, 12)   numpy array
    }

GNN note: GNNs take graph object lists rather than numpy arrays, so they
don't plug into this X_dict interface without a thin adapter layer.
The ensembles here target FingerprintModel (lgbm_morgan, lgbm_descriptors)
combinations; GNN ensembling is left for a future iteration.
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
# Internal helper
# ---------------------------------------------------------------------------

def _get_predictions(model: Any, X: np.ndarray, task_type: str) -> np.ndarray:
    """Call the right predict method and return a 1-D array.

    For regression: model.predict(X).
    For classification: model.predict_proba(X)[:, 1] if available, else
    model.predict(X).  Probabilities are needed for soft voting / meta-learners.
    """
    if task_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()
    return np.asarray(model.predict(X)).ravel()


# ---------------------------------------------------------------------------
# VotingEnsemble
# ---------------------------------------------------------------------------

class VotingEnsemble:
    """Average predictions from multiple base models.

    The simplest possible ensemble: each base model votes with its prediction
    and the final output is a (optionally weighted) mean.  Works well when:

    - Base models have low prediction correlation (e.g., fingerprints and
      descriptors encode orthogonal aspects of the molecule).
    - Models are similarly calibrated (all on the same output scale).

    For regression: weighted mean of predictions.
    For classification: weighted mean of predicted probabilities, then
    threshold at 0.5 for hard labels.

    Parameters
    ----------
    models : dict[str, Any]
        Unfitted model instances keyed by name.
        Example: {"lgbm_morgan": FingerprintModel(...), "lgbm_descriptors": FingerprintModel(...)}
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

    def fit(self, X_dict: dict[str, np.ndarray], y: np.ndarray) -> "VotingEnsemble":
        """Train all base models on the same target vector.

        Parameters
        ----------
        X_dict : dict[str, np.ndarray]
            Feature matrix for each model. ``X_dict[name]`` is passed to
            ``models[name].fit(X, y)``.
        y : np.ndarray
            Target values.
        """
        for name, model in self.models.items():
            if name not in X_dict:
                raise KeyError(
                    f"Model '{name}' not found in X_dict. "
                    f"Available keys: {list(X_dict.keys())}"
                )
            logger.info(f"  Fitting base model: {name} ({X_dict[name].shape[0]} samples)")
            model.fit(X_dict[name], y)

        self._fitted = True
        logger.info(f"VotingEnsemble fitted ({len(self.models)} base models)")
        return self

    def predict(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Return ensemble predictions.

        Regression: weighted mean of base model predictions.
        Classification: weighted mean of predicted probabilities, hard-thresholded.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        all_preds = [
            _get_predictions(model, X_dict[name], self.task_type)
            for name, model in self.models.items()
        ]

        if self.weights is not None:
            avg = np.average(all_preds, axis=0, weights=self.weights)
        else:
            avg = np.mean(all_preds, axis=0)

        if self.task_type == "classification":
            return (avg >= 0.5).astype(int)
        return avg

    def predict_proba(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Return averaged predicted probabilities (classification only)."""
        all_preds = [
            _get_predictions(model, X_dict[name], "classification")
            for name, model in self.models.items()
        ]
        if self.weights is not None:
            return np.average(all_preds, axis=0, weights=self.weights)
        return np.mean(all_preds, axis=0)

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

    Unlike simple voting, blending learns optimal combination weights from
    a held-out validation set.  The meta-model (Ridge for regression,
    LogisticRegression for classification) maps base model predictions to
    the final output.

    Training procedure:
    1. Fit all base models on ``X_dict_train`` / ``y_train``.
    2. Generate validation predictions from each base model → meta-features.
    3. Fit the meta-model: meta-features → ``y_val``.

    At inference, base model predictions are fed through the fitted meta-model.

    The meta-model's learned coefficients reflect each base model's contribution.
    Call ``get_blend_weights()`` to inspect them after fitting.

    Parameters
    ----------
    models : dict[str, Any]
        Unfitted model instances keyed by name.
    task_type : str
        "regression" or "classification".
    meta_alpha : float
        Ridge regularization (regression) or inverse LogisticRegression C
        (classification). Higher = more regularized = weights closer to equal.
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
        X_dict_train: dict[str, np.ndarray],
        y_train: np.ndarray,
        X_dict_val: dict[str, np.ndarray],
        y_val: np.ndarray,
    ) -> "BlendingEnsemble":
        """Fit base models on train; learn blend weights from val predictions.

        Parameters
        ----------
        X_dict_train : dict[str, np.ndarray]
            Training features for each model.
        y_train : np.ndarray
            Training targets.
        X_dict_val : dict[str, np.ndarray]
            Validation features (used to generate meta-learner training data).
        y_val : np.ndarray
            Validation targets.
        """
        # Step 1: Train base models on training set
        for name, model in self.models.items():
            logger.info(f"  Fitting base model: {name} ({X_dict_train[name].shape[0]} samples)")
            model.fit(X_dict_train[name], y_train)

        # Step 2: Generate meta-features from validation predictions
        val_meta_X = self._make_meta_features(X_dict_val)
        logger.info(f"  Meta-features shape: {val_meta_X.shape}")

        # Step 3: Fit meta-model on val predictions → val targets
        if self.task_type == "classification":
            self.meta_model = LogisticRegression(
                C=1.0 / self.meta_alpha, max_iter=500, solver="lbfgs"
            )
        else:
            self.meta_model = Ridge(alpha=self.meta_alpha)

        self.meta_model.fit(val_meta_X, y_val)

        weights = self.get_blend_weights()
        logger.info(f"  Blend weights: {weights}")

        self._fitted = True
        logger.info(f"BlendingEnsemble fitted ({len(self.models)} base models)")
        return self

    def predict(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Return blended predictions via the meta-model."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        preds = self.meta_model.predict(self._make_meta_features(X_dict))
        if self.task_type == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def predict_proba(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
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

    def _make_meta_features(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Stack base model predictions into a (N, n_models) matrix."""
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
    """K-fold stacking with a meta-learner trained on out-of-fold predictions.

    More data-efficient than blending: instead of holding out a dedicated
    validation set, every training sample contributes to the meta-features
    via K-fold cross-validation.

    Training procedure:
    1. For each of K folds:
       a. Deep-copy each base model (fresh instance, unfitted).
       b. Train the fold copies on the (K-1) in-fold rows.
       c. Predict on the held-out fold → out-of-fold (OOF) predictions.
    2. Stack OOF predictions → meta-feature matrix of shape (N_train, n_models).
    3. Fit the meta-learner on OOF features → training targets.
    4. Re-fit all base models on the full training set for inference.

    At inference: base models predict → meta-learner combines.

    Parameters
    ----------
    models : dict[str, Any]
        Unfitted model instances keyed by name.  Each is deep-copied per fold;
        the originals are re-fitted on the full dataset at the end of ``fit()``.
    task_type : str
        "regression" or "classification".
    n_folds : int
        Number of CV folds for OOF generation (default 5).
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

    def fit(self, X_dict: dict[str, np.ndarray], y: np.ndarray) -> "StackingEnsemble":
        """Generate OOF predictions, fit meta-learner, refit base models.

        Parameters
        ----------
        X_dict : dict[str, np.ndarray]
            Feature matrices for each base model.
        y : np.ndarray
            Training targets.
        """
        n = len(y)
        model_names = list(self.models.keys())
        oof_preds = np.zeros((n, len(model_names)))

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        logger.info(
            f"Generating OOF predictions "
            f"({self.n_folds} folds × {len(model_names)} models)..."
        )
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(y)):
            y_fold_train = y[train_idx]
            for col_idx, name in enumerate(model_names):
                fold_model = copy.deepcopy(self.models[name])
                fold_model.fit(X_dict[name][train_idx], y_fold_train)
                oof_preds[val_idx, col_idx] = _get_predictions(
                    fold_model, X_dict[name][val_idx], self.task_type
                )
            logger.info(f"  Fold {fold_idx + 1}/{self.n_folds} complete")

        # Fit meta-learner on OOF predictions
        if self.task_type == "classification":
            self.meta_model = LogisticRegression(
                C=1.0 / self.meta_alpha, max_iter=500, solver="lbfgs"
            )
        else:
            self.meta_model = Ridge(alpha=self.meta_alpha)

        self.meta_model.fit(oof_preds, y)
        logger.info(f"  Meta-learner fitted on OOF features {oof_preds.shape}")
        logger.info(f"  Meta-model importances: {self.get_meta_importances()}")

        # Refit base models on the full dataset for inference
        logger.info("Re-fitting base models on full training set...")
        for name, model in self.models.items():
            model.fit(X_dict[name], y)
            logger.info(f"  Re-fitted: {name}")

        self._fitted = True
        logger.info("StackingEnsemble fitted")
        return self

    def predict(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Return stacked predictions via the meta-learner."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        preds = self.meta_model.predict(self._make_meta_features(X_dict))
        if self.task_type == "classification":
            return (preds >= 0.5).astype(int)
        return preds

    def predict_proba(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
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
        return dict(zip(self.models.keys(), coef.flat))

    def _make_meta_features(self, X_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Stack base model predictions into a (N, n_models) matrix."""
        cols = [
            _get_predictions(model, X_dict[name], self.task_type)
            for name, model in self.models.items()
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

    # Build feature matrices for each split
    X = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        smiles = split_df["smiles"].tolist()
        X[f"{split_name}_morgan"] = compute_fingerprints(smiles)
        X[f"{split_name}_desc"] = compute_descriptors(smiles).values

    y_train = train_df["y"].values
    y_val = val_df["y"].values
    y_test = test_df["y"].values

    X_train = {"lgbm_morgan": X["train_morgan"], "lgbm_descriptors": X["train_desc"]}
    X_val   = {"lgbm_morgan": X["val_morgan"],   "lgbm_descriptors": X["val_desc"]}
    X_test  = {"lgbm_morgan": X["test_morgan"],  "lgbm_descriptors": X["test_desc"]}

    def make_base_models():
        return {
            "lgbm_morgan": create_model("lgbm_morgan", task_type="regression"),
            "lgbm_descriptors": create_model("lgbm_descriptors", task_type="regression"),
        }

    print("\n--- Individual base models ---")
    for name in ["lgbm_morgan", "lgbm_descriptors"]:
        m = create_model(name, task_type="regression")
        key = "lgbm_morgan" if name == "lgbm_morgan" else "lgbm_descriptors"
        train_key = "train_morgan" if name == "lgbm_morgan" else "train_desc"
        test_key = "test_morgan" if name == "lgbm_morgan" else "test_desc"
        m.fit(X[train_key], y_train)
        metrics = compute_metrics(y_test, m.predict(X[test_key]), task_type="regression")
        print(f"  {name}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    print("\n--- VotingEnsemble ---")
    voting = VotingEnsemble(make_base_models(), task_type="regression")
    voting.fit(X_train, y_train)
    metrics = compute_metrics(y_test, voting.predict(X_test), task_type="regression")
    print(f"  MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    print("\n--- BlendingEnsemble ---")
    blending = BlendingEnsemble(make_base_models(), task_type="regression", meta_alpha=1.0)
    blending.fit(X_train, y_train, X_val, y_val)
    print(f"  Blend weights: {blending.get_blend_weights()}")
    metrics = compute_metrics(y_test, blending.predict(X_test), task_type="regression")
    print(f"  MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    print("\n--- StackingEnsemble ---")
    stacking = StackingEnsemble(make_base_models(), task_type="regression", n_folds=5)
    stacking.fit(X_train, y_train)
    print(f"  Meta importances: {stacking.get_meta_importances()}")
    metrics = compute_metrics(y_test, stacking.predict(X_test), task_type="regression")
    print(f"  MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

    import IPython
    IPython.embed()
