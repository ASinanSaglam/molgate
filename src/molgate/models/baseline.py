"""Fingerprint-based baseline models — LightGBM on Morgan FPs or descriptors.

This module wraps gradient-boosted tree models (LightGBM, optionally XGBoost)
behind a unified interface.  The key design decisions:

1. **Unified interface**: ``fit(X, y)`` and ``predict(X)`` work the same
   regardless of whether the input is Morgan fingerprints or RDKit descriptors.
   This makes the model interchangeable in training loops and evaluation code.

2. **Task-aware**: The model automatically picks a regressor or classifier
   based on the ``task_type`` parameter ("regression" or "classification").

3. **Optuna integration**: ``tune(X, y, n_trials)`` runs Bayesian hyperparameter
   optimisation with cross-validation.  Optuna suggests LightGBM hyperparameters,
   trains with CV, and returns the best configuration.  This replaces hand-tuning.

4. **No featurization inside the model**: The model receives pre-computed
   feature matrices (from ``data/featurizer.py``).  Keeping featurization
   separate means the same model class works with any feature type.

Why LightGBM over XGBoost?
  Both are competitive, but LightGBM is faster on sparse features (fingerprints
  are >95% zeros) due to its histogram-based splitting and leaf-wise growth.
  XGBoost support is included as a fallback via the ``estimator`` parameter.
"""

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


# Suppress Optuna's verbose logging (we log results ourselves)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class FingerprintModel:
    """Gradient-boosted tree model for molecular property prediction.

    Wraps LightGBM (or XGBoost) with a scikit-learn-compatible interface.
    Handles both regression and classification based on task_type.

    Args:
        task_type: "regression" or "classification".
        estimator: "lightgbm" or "xgboost".
        params: Dict of model-specific hyperparameters.  Passed directly
            to the underlying estimator constructor.  If None, uses
            sensible defaults.

    Example::

        from molgate.data.featurizer import compute_fingerprints
        from molgate.models.baseline import FingerprintModel

        X_train = compute_fingerprints(train_smiles)
        model = FingerprintModel(task_type="regression")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    """

    def __init__(
        self,
        task_type: str = "regression",
        estimator: str = "lightgbm",
        params: dict[str, Any] | None = None,
    ):
        if task_type not in ("regression", "classification"):
            raise ValueError(f"task_type must be 'regression' or 'classification', got '{task_type}'")
        if estimator not in ("lightgbm", "xgboost"):
            raise ValueError(f"estimator must be 'lightgbm' or 'xgboost', got '{estimator}'")

        self.task_type = task_type
        self.estimator_name = estimator
        self.params = params or {}
        self.model_ = None  # Set after fit()
        self._build_model()

    def _build_model(self) -> None:
        """Instantiate the underlying estimator with current params."""
        if self.estimator_name == "lightgbm":
            if self.task_type == "regression":
                self.model_ = lgb.LGBMRegressor(**self.params)
            else:
                self.model_ = lgb.LGBMClassifier(**self.params)
        elif self.estimator_name == "xgboost":
            import xgboost as xgb
            if self.task_type == "regression":
                self.model_ = xgb.XGBRegressor(**self.params)
            else:
                self.model_ = xgb.XGBClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FingerprintModel":
        """Train the model on feature matrix X and targets y.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
                Morgan fingerprints (n_features=2048) or descriptors (n_features=12).
            y: Target values of shape (n_samples,).
                Continuous for regression, binary {0,1} for classification.

        Returns:
            self (for method chaining).
        """
        logger.info(
            f"Fitting {self.estimator_name} ({self.task_type}) on "
            f"{X.shape[0]} samples, {X.shape[1]} features"
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for feature matrix X.

        For regression, returns continuous predictions.
        For classification, returns predicted probabilities of the positive class
        (not hard labels) — this is needed for AUROC computation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        if self.task_type == "classification":
            # Return probability of positive class for AUROC
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities for classification.

        Returns the full (n_samples, n_classes) probability matrix from
        the underlying estimator.  For binary classification col[:, 1] is
        the positive-class probability.  Raises if called on a regression model.
        """
        if self.task_type != "classification":
            raise RuntimeError("predict_proba() is only valid for classification models.")
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.model_.predict_proba(X)

    def get_params(self) -> dict:
        """Return the model's current hyperparameters.

        Useful for W&B logging — records exactly what was trained.
        """
        return {
            "task_type": self.task_type,
            "estimator": self.estimator_name,
            **self.model_.get_params(),
        }

    def feature_importances(self) -> np.ndarray | None:
        """Return feature importances from the trained model.

        LightGBM and XGBoost both provide feature importances based on
        split gain (how much each feature contributes to reducing the loss).

        Returns:
            Array of shape (n_features,), or None if not fitted.
        """
        if self.model_ is None:
            return None
        return self.model_.feature_importances_

    # ------------------------------------------------------------------
    # Hyperparameter tuning with Optuna
    # ------------------------------------------------------------------

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        cv_folds: int = 5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run Optuna hyperparameter search with cross-validation.

        Searches over LightGBM-specific hyperparameters using Bayesian
        optimisation (TPE sampler).  Each trial trains with k-fold CV
        and reports the mean validation metric.

        After tuning, the model is rebuilt with the best parameters and
        refitted on the full training data.

        Args:
            X: Training feature matrix.
            y: Training targets.
            n_trials: Number of Optuna trials (more = better, but slower).
            cv_folds: Number of cross-validation folds.
            seed: Random seed for reproducibility.

        Returns:
            Dict of best hyperparameters found.
        """
        scoring = "neg_root_mean_squared_error" if self.task_type == "regression" else "roc_auc"

        def objective(trial: optuna.Trial) -> float:
            trial_params = self._suggest_params(trial, seed)
            model = self._make_estimator(trial_params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            return scores.mean()

        study = optuna.create_study(
            direction="maximize",  # Both neg_RMSE and AUROC: higher is better
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials)

        best_params = self._suggest_to_full_params(study.best_params, seed)
        logger.info(
            f"Optuna tuning complete: best score={study.best_value:.4f} "
            f"({n_trials} trials)"
        )

        # Rebuild model with best params and fit on full data
        self.params = best_params
        self._build_model()
        self.fit(X, y)

        return best_params

    def _suggest_params(self, trial: optuna.Trial, seed: int) -> dict:
        """Define the Optuna search space for LightGBM."""
        if self.estimator_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbose": -1,
                "random_state": seed,
            }
        else:
            # XGBoost search space
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "verbosity": 0,
                "random_state": seed,
            }

    def _suggest_to_full_params(self, best_trial_params: dict, seed: int) -> dict:
        """Convert Optuna's best trial params to full estimator params."""
        params = dict(best_trial_params)
        if self.estimator_name == "lightgbm":
            params["verbose"] = -1
            params["random_state"] = seed
        else:
            params["verbosity"] = 0
            params["random_state"] = seed
        return params

    def _make_estimator(self, params: dict):
        """Create a fresh estimator instance with given params."""
        if self.estimator_name == "lightgbm":
            if self.task_type == "regression":
                return lgb.LGBMRegressor(**params)
            return lgb.LGBMClassifier(**params)
        else:
            import xgboost as xgb
            if self.task_type == "regression":
                return xgb.XGBRegressor(**params)
            return xgb.XGBClassifier(**params)


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.featurizer import compute_descriptors, compute_fingerprints
    from molgate.data.loaders import load_dataset
    from molgate.data.splits import random_split

    df = load_dataset("solubility")
    splits = random_split(df, seed=42)
    train_df, test_df = splits["train"], splits["test"]

    # --- Morgan fingerprints ---
    X_train_fp = compute_fingerprints(train_df["smiles"].tolist())
    X_test_fp = compute_fingerprints(test_df["smiles"].tolist())
    y_train = train_df["y"].values
    y_test = test_df["y"].values

    print("=== LightGBM + Morgan Fingerprints ===")
    model_fp = FingerprintModel(
        task_type="regression",
        params={"n_estimators": 500, "learning_rate": 0.05, "verbose": -1},
    )
    model_fp.fit(X_train_fp, y_train)
    preds_fp = model_fp.predict(X_test_fp)
    rmse_fp = np.sqrt(np.mean((preds_fp - y_test) ** 2))
    print(f"  RMSE: {rmse_fp:.4f}")

    # --- RDKit descriptors ---
    X_train_desc = compute_descriptors(train_df["smiles"].tolist()).values
    X_test_desc = compute_descriptors(test_df["smiles"].tolist()).values

    print("\n=== LightGBM + RDKit Descriptors ===")
    model_desc = FingerprintModel(
        task_type="regression",
        params={"n_estimators": 500, "learning_rate": 0.05, "verbose": -1},
    )
    model_desc.fit(X_train_desc, y_train)
    preds_desc = model_desc.predict(X_test_desc)
    rmse_desc = np.sqrt(np.mean((preds_desc - y_test) ** 2))
    print(f"  RMSE: {rmse_desc:.4f}")

    print(f"\n  FP features: {X_train_fp.shape[1]}, Desc features: {X_train_desc.shape[1]}")
    print(f"  FP RMSE: {rmse_fp:.4f}, Desc RMSE: {rmse_desc:.4f}")

    import IPython; IPython.embed()
