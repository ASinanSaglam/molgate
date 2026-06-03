"""Sklearn-compatible wrapper for Ridge, RandomForest, SVR, and similar estimators.

Why a separate module?
    FingerprintModel (baseline.py) handles LightGBM and XGBoost — estimators that
    need gradient-boosting-specific wiring (early stopping, verbose, etc.).
    Sklearn estimators (Ridge, RF, SVR) have a simpler uniform API and don't
    need that wiring.  SklearnModel gives them the same interface as
    FingerprintModel so they plug into ensembles and the factory without
    special-casing.

Supported estimators (task_type-aware):
    Regression:     Ridge, Lasso, ElasticNet, RandomForestRegressor, SVR
    Classification: LogisticRegression, RandomForestClassifier, SVC
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


# Mapping from YAML class name → sklearn class.
# Importing from here avoids repeated string lookups across the codebase.
_SKLEARN_CLASSES: dict[str, tuple[str, str]] = {
    # (module_path, class_name)
    "Ridge":                    ("sklearn.linear_model",  "Ridge"),
    "Lasso":                    ("sklearn.linear_model",  "Lasso"),
    "ElasticNet":               ("sklearn.linear_model",  "ElasticNet"),
    "LogisticRegression":       ("sklearn.linear_model",  "LogisticRegression"),
    "RandomForestRegressor":    ("sklearn.ensemble",      "RandomForestRegressor"),
    "RandomForestClassifier":   ("sklearn.ensemble",      "RandomForestClassifier"),
    "GradientBoostingRegressor":("sklearn.ensemble",      "GradientBoostingRegressor"),
    "GradientBoostingClassifier":("sklearn.ensemble",     "GradientBoostingClassifier"),
    "SVR":                      ("sklearn.svm",           "SVR"),
    "SVC":                      ("sklearn.svm",           "SVC"),
    "KNeighborsRegressor":      ("sklearn.neighbors",     "KNeighborsRegressor"),
    "KNeighborsClassifier":     ("sklearn.neighbors",     "KNeighborsClassifier"),
}


def get_sklearn_class(class_name: str):
    """Import and return a sklearn estimator class by name.

    Args:
        class_name: Name from _SKLEARN_CLASSES (e.g., "Ridge", "SVR").

    Returns:
        The estimator class (not an instance).

    Raises:
        ValueError: If the name is not in the registry.
    """
    if class_name not in _SKLEARN_CLASSES:
        valid = ", ".join(sorted(_SKLEARN_CLASSES.keys()))
        raise ValueError(
            f"Unknown sklearn class: {class_name!r}. "
            f"Supported: {valid}"
        )
    module_path, cls_name = _SKLEARN_CLASSES[class_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


class SklearnModel:
    """Thin wrapper making sklearn estimators compatible with the molgate model interface.

    Provides ``fit(X, y)``, ``predict(X)``, ``predict_proba(X)``, and ``get_params()``
    with the same semantics as FingerprintModel.  Attaches ``feature_type``,
    ``fp_radius``, and ``fp_nbits`` metadata so training flows know which
    featurization to compute.

    Args:
        estimator: A fitted or unfitted sklearn estimator instance.
        task_type: "regression" or "classification".
        feature_type: "morgan" or "descriptors" — determines which featurizer
            the training flow uses.  Stored as metadata, not used internally.
        fp_radius: Morgan FP radius (metadata only; ignored for descriptors).
        fp_nbits: Morgan FP bit length (metadata only; ignored for descriptors).

    Example::

        from sklearn.linear_model import Ridge
        from molgate.models.sklearn_models import SklearnModel

        model = SklearnModel(Ridge(alpha=1.0), task_type="regression",
                             feature_type="descriptors")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        estimator: Any,
        task_type: str = "regression",
        feature_type: str = "descriptors",
        fp_radius: int = 2,
        fp_nbits: int = 2048,
    ):
        if task_type not in ("regression", "classification"):
            raise ValueError(
                f"task_type must be 'regression' or 'classification', got {task_type!r}"
            )
        self.estimator = estimator
        self.task_type = task_type
        # Featurization metadata — consumed by training flows, not by this class.
        self.feature_type = feature_type
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnModel":
        """Train the estimator.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Targets (n_samples,). Continuous for regression, binary for classification.

        Returns:
            self (for method chaining).
        """
        logger.info(
            f"Fitting {type(self.estimator).__name__} ({self.task_type}) on "
            f"{X.shape[0]} samples, {X.shape[1]} features"
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions.

        Regression: continuous values.
        Classification: probability of the positive class ([:, 1] from predict_proba
        if available, else decision_function scores normalized to [0, 1]).
        """
        if self.task_type == "classification":
            if hasattr(self.estimator, "predict_proba"):
                return self.estimator.predict_proba(X)[:, 1]
            if hasattr(self.estimator, "decision_function"):
                scores = self.estimator.decision_function(X)
                # Sigmoid to map decision scores to [0, 1]
                return 1.0 / (1.0 + np.exp(-scores))
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return full probability matrix (n_samples, n_classes).

        For binary classification use [:, 1] for the positive-class probability.
        Raises if the estimator doesn't support probability estimates or if
        this is a regression model.
        """
        if self.task_type != "classification":
            raise RuntimeError("predict_proba() is only valid for classification models.")
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        raise RuntimeError(
            f"{type(self.estimator).__name__} does not support predict_proba(). "
            "Use SVC(probability=True) or a probabilistic estimator."
        )

    def get_params(self) -> dict[str, Any]:
        """Return estimator hyperparameters plus task metadata."""
        params = {
            "task_type": self.task_type,
            "estimator_class": type(self.estimator).__name__,
            "feature_type": self.feature_type,
        }
        if hasattr(self.estimator, "get_params"):
            params.update(self.estimator.get_params())
        return params

    # ------------------------------------------------------------------
    # Hyperparameter tuning with Optuna
    # ------------------------------------------------------------------

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 30,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run Optuna hyperparameter search with cross-validation.

        Searches an estimator-specific hyperparameter space using Bayesian
        optimisation (TPE sampler).  Each trial trains with K-fold CV and
        reports the mean validation score.

        After tuning, the estimator is recreated with the best parameters
        and refitted on the full training data.

        Supported estimators and their search spaces:

        ============== =====================================================
        Ridge / Lasso  alpha [1e-3, 100] log-uniform
        ElasticNet     alpha [1e-3, 100] log-uniform; l1_ratio [0.0, 1.0]
        RF             n_estimators, max_depth, min_samples_leaf, max_features
        GBM            n_estimators, lr, max_depth, subsample, min_samples_leaf
        SVR / SVC      C [1e-2, 1e3] log-uniform; epsilon [1e-3, 1.0]; kernel
        Logistic       C [1e-3, 100] log-uniform
        KNN            n_neighbors [1, 30]; weights
        ============== =====================================================

        Args:
            X: Training feature matrix (n_samples, n_features).
            y: Training targets.
            n_trials: Number of Optuna trials (default 30 — faster than LGBM's 50
                because sklearn estimators train more quickly).
            cv_folds: Number of cross-validation folds.
            seed: Random seed for reproducibility.

        Returns:
            Dict of best hyperparameters found.
        """
        scoring = "neg_root_mean_squared_error" if self.task_type == "regression" else "roc_auc"

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, seed)
            estimator = type(self.estimator)(**params)
            scores = cross_val_score(estimator, X, y, cv=cv_folds, scoring=scoring)
            return float(scores.mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(
            f"Optuna tuning complete ({type(self.estimator).__name__}): "
            f"best score={study.best_value:.4f} over {n_trials} trials"
        )

        # Rebuild estimator with best params and refit on full data
        self.estimator = type(self.estimator)(**best_params)
        self.fit(X, y)

        return best_params

    def _suggest_params(self, trial: optuna.Trial, seed: int) -> dict[str, Any]:
        """Return a hyperparameter dict for the trial based on estimator class."""
        cls_name = type(self.estimator).__name__

        if cls_name == "Ridge":
            return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}

        if cls_name == "Lasso":
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
                "max_iter": 2000,
            }

        if cls_name == "ElasticNet":
            return {
                "alpha":    trial.suggest_float("alpha", 1e-3, 100.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": 2000,
            }

        if cls_name == "LogisticRegression":
            return {
                "C":        trial.suggest_float("C", 1e-3, 100.0, log=True),
                "max_iter": 500,
                "solver":   "lbfgs",
            }

        if cls_name in ("RandomForestRegressor", "RandomForestClassifier"):
            return {
                "n_estimators":    trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth":       trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features":    trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5]
                ),
                "n_jobs": -1,
                "random_state": seed,
            }

        if cls_name in ("GradientBoostingRegressor", "GradientBoostingClassifier"):
            return {
                "n_estimators":    trial.suggest_int("n_estimators", 50, 500, step=50),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth":       trial.suggest_int("max_depth", 2, 8),
                "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 20),
                "random_state":    seed,
            }

        if cls_name == "SVR":
            return {
                "C":       trial.suggest_float("C", 1e-2, 1e3, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                "kernel":  trial.suggest_categorical("kernel", ["rbf", "linear"]),
            }

        if cls_name == "SVC":
            return {
                "C":      trial.suggest_float("C", 1e-2, 1e3, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
                "probability": True,
            }

        if cls_name in ("KNeighborsRegressor", "KNeighborsClassifier"):
            return {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
                "weights":     trial.suggest_categorical("weights", ["uniform", "distance"]),
            }

        raise ValueError(
            f"No search space defined for estimator: {cls_name!r}. "
            f"Add it to SklearnModel._suggest_params() or call tune() after "
            f"subclassing."
        )

    def __repr__(self) -> str:
        return (
            f"SklearnModel({type(self.estimator).__name__}, "
            f"task_type={self.task_type!r}, feature_type={self.feature_type!r})"
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
    from molgate.training.metrics import compute_metrics
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR

    print("Loading solubility dataset...")
    df = load_dataset("solubility")
    splits = random_split(df, seed=42)
    train_df, test_df = splits["train"], splits["test"]

    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    y_train = train_df["y"].values
    y_test = test_df["y"].values

    X_desc_train = compute_descriptors(train_smiles).values
    X_desc_test = compute_descriptors(test_smiles).values
    X_fp_train = compute_fingerprints(train_smiles)
    X_fp_test = compute_fingerprints(test_smiles)

    results = []
    for name, model_cls, X_tr, X_te, params in [
        ("Ridge (descriptors)",   Ridge,                    X_desc_train, X_desc_test, {"alpha": 1.0}),
        ("RF (morgan)",           RandomForestRegressor,    X_fp_train,   X_fp_test,   {"n_estimators": 200, "n_jobs": -1, "random_state": 42}),
        ("RF (descriptors)",      RandomForestRegressor,    X_desc_train, X_desc_test, {"n_estimators": 200, "n_jobs": -1, "random_state": 42}),
        ("SVR (descriptors)",     SVR,                      X_desc_train, X_desc_test, {"kernel": "rbf", "C": 10.0}),
    ]:
        m = SklearnModel(model_cls(**params), task_type="regression", feature_type="descriptors")
        m.fit(X_tr, y_train)
        metrics = compute_metrics(y_test, m.predict(X_te), task_type="regression")
        results.append((name, metrics))
        print(f"  {name:<30s}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

    import IPython
    IPython.embed()
