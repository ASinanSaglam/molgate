"""Model factory: create models from config dictionaries.

This module provides a single entry point — ``create_model`` — that takes
a model configuration (from configs/models.yaml or a plain dict) and returns
an instantiated model ready for training.

The factory pattern decouples "which model to use" from "how to build it."
Prefect flows and the CLI only need to know the model name (e.g., "lgbm_morgan");
the factory handles the class imports, parameter mapping, and validation.

Supported model types:
    - "fingerprint" → FingerprintModel (LightGBM or XGBoost on fingerprints/descriptors)
    - "gnn"         → MoleculeGNN (message-passing GNN on molecular graphs)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_models_config(config_path: str | Path | None = None) -> dict:
    """Load the models section from a YAML config file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML file. Defaults to ``configs/models.yaml``
        relative to the project root.

    Returns
    -------
    dict
        The ``models:`` section of the config, keyed by model name.
    """
    if config_path is None:
        # Walk up from this file to find the project root (where configs/ lives)
        here = Path(__file__).resolve().parent       # src/molgate/models/
        project_root = here.parent.parent.parent     # molgate/
        config_path = project_root / "configs" / "models.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Models config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return raw.get("models", raw)


def create_model(
    model_name: str,
    task_type: str = "regression",
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    base_models: dict[str, Any] | None = None,
):
    """Create a model instance from a named configuration.

    Parameters
    ----------
    model_name : str
        Name of the model in the config (e.g., "lgbm_morgan", "gnn").
    task_type : str
        "regression" or "classification". Passed to the model constructor.
    config_path : str or Path, optional
        Path to models.yaml. Defaults to ``configs/models.yaml``.
    overrides : dict, optional
        Key-value pairs that override config values. Useful for
        hyperparameter sweeps without modifying the YAML file.
    base_models : dict[str, model], optional
        Pre-instantiated base model objects for ensemble configs.  When
        provided, the config's ``base_models`` name list is ignored and
        these instances are used directly.  Keys are the model names that
        ``X_dict`` will use when calling ``fit`` / ``predict``.  Ignored
        for non-ensemble model types.

    Returns
    -------
    model
        An instantiated model (FingerprintModel or MoleculeGNN).

    Raises
    ------
    KeyError
        If ``model_name`` is not found in the config.
    ValueError
        If the model ``type`` field is unknown.

    Examples
    --------
    >>> model = create_model("lgbm_morgan", task_type="regression")
    >>> model = create_model("gnn", task_type="classification")
    >>> model = create_model("lgbm_morgan", overrides={"lightgbm_params": {"n_estimators": 100}})
    >>> # Inject pre-built base models into an ensemble:
    >>> ensemble = create_model("ensemble_stacking", base_models=fitted_base_models)
    """
    models_config = _load_models_config(config_path)

    if model_name not in models_config:
        available = ", ".join(sorted(models_config.keys()))
        raise KeyError(
            f"Unknown model name: {model_name!r}. "
            f"Available models: {available}"
        )

    # Deep-copy to avoid mutating the loaded config
    config = _deep_merge(dict(models_config[model_name]), overrides or {})
    model_type = config.pop("type")

    logger.info(f"Creating model {model_name!r} (type={model_type}, task={task_type})")

    if model_type == "fingerprint":
        return _create_fingerprint_model(config, task_type)
    elif model_type == "gnn":
        return _create_gnn_model(config, task_type)
    elif model_type == "sklearn":
        return _create_sklearn_model(config, task_type)
    elif model_type == "ensemble":
        return _create_ensemble_model(config, task_type, config_path, base_models=base_models)
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Supported types: 'fingerprint', 'gnn', 'sklearn', 'ensemble'"
        )


def _create_fingerprint_model(config: dict, task_type: str):
    """Build a FingerprintModel from config.

    Config keys consumed:
        - estimator: "lightgbm" or "xgboost"
        - lightgbm_params / xgboost_params: passed to the estimator
        - fingerprint, fp_radius, fp_nbits: stored as metadata (used by
          the training flow to decide which featurization to apply)
    """
    from molgate.models.baseline import FingerprintModel

    estimator = config.get("estimator", "lightgbm")
    params = config.get(f"{estimator}_params", config.get("lightgbm_params", {}))

    model = FingerprintModel(
        task_type=task_type,
        estimator=estimator,
        params=params,
    )

    # Attach featurization metadata so the training flow knows what to compute.
    # These aren't used by FingerprintModel itself — it just takes X arrays —
    # but the flow needs to know "should I compute Morgan FPs or descriptors?"
    model.feature_type = config.get("fingerprint", "morgan")
    model.fp_radius = config.get("fp_radius", 2)
    model.fp_nbits = config.get("fp_nbits", 2048)

    return model


def _create_gnn_model(config: dict, task_type: str):
    """Build a MoleculeGNN from config.

    Config keys consumed:
        - hidden_dim, num_layers, dropout, pool: GNN architecture params
        - training: nested dict with lr, epochs, patience, etc. — stored
          as metadata for the trainer, not used by the model itself.
    """
    from molgate.models.gnn import MoleculeGNN

    # Separate architecture params from training params
    training_config = config.pop("training", {})

    model = MoleculeGNN(
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 3),
        task_type=task_type,
        dropout=config.get("dropout", 0.1),
        pool=config.get("pool", "mean"),
    )

    # Attach training config as metadata for the trainer
    model.training_config = training_config

    return model


def _create_sklearn_model(config: dict, task_type: str):
    """Build a SklearnModel from config.

    Config keys consumed:
        - sklearn_class: name of the sklearn estimator (e.g. "Ridge", "SVR")
        - sklearn_params: kwargs passed to the estimator constructor
        - fingerprint / fp_radius / fp_nbits: featurization metadata
    """
    from molgate.models.sklearn_models import SklearnModel, get_sklearn_class

    cls_name = config.get("sklearn_class")
    if not cls_name:
        raise ValueError("sklearn model config must include 'sklearn_class'")

    params = config.get("sklearn_params", {})
    estimator_cls = get_sklearn_class(cls_name)

    # Adjust for classification — some sklearn classes accept class_weight etc.
    # For now just pass params as-is; factory caller sets task_type.
    estimator = estimator_cls(**params)

    feature_type = config.get("fingerprint", "descriptors")
    model = SklearnModel(
        estimator=estimator,
        task_type=task_type,
        feature_type=feature_type,
        fp_radius=config.get("fp_radius", 2),
        fp_nbits=config.get("fp_nbits", 2048),
    )
    return model


def _create_ensemble_model(
    config: dict,
    task_type: str,
    config_path=None,
    base_models: dict[str, Any] | None = None,
):
    """Build an ensemble model from config.

    Config keys consumed:
        - ensemble_type: "voting", "blending", or "stacking"
        - base_models: list of model names (resolved recursively via create_model)
        - weights: optional list of floats (voting only)
        - meta_alpha: float (blending / stacking)
        - n_folds: int (stacking)

    GNN base models are automatically wrapped in GNNModelWrapper.

    If ``base_models`` is provided as a dict of pre-instantiated model objects,
    the config's ``base_models`` name list is skipped entirely.
    """
    from molgate.models.ensemble import (
        BlendingEnsemble,
        GNNModelWrapper,
        StackingEnsemble,
        VotingEnsemble,
    )

    ensemble_type = config.get("ensemble_type", "voting")

    if base_models is not None:
        # Caller supplied pre-built instances — wrap any bare GNNs and use as-is.
        # Check for GNNModelWrapper first to avoid double-wrapping (it also has
        # training_config set on it).
        resolved: dict[str, Any] = {}
        for name, m in base_models.items():
            if hasattr(m, "training_config") and not isinstance(m, GNNModelWrapper):
                training_cfg = m.training_config
                m = GNNModelWrapper(
                    gnn_model=m,
                    training_config=training_cfg,
                    task_type=task_type,
                )
            resolved[name] = m
    else:
        base_model_names = config.get("base_models", [])
        if not base_model_names:
            raise ValueError("ensemble config must include at least one entry in 'base_models'")

        resolved = {}
        for name in base_model_names:
            m = create_model(name, task_type=task_type, config_path=config_path)
            if hasattr(m, "training_config"):
                training_cfg = m.training_config
                m = GNNModelWrapper(
                    gnn_model=m,
                    training_config=training_cfg,
                    task_type=task_type,
                )
            resolved[name] = m

    meta_alpha = config.get("meta_alpha", 1.0)
    n_folds = config.get("n_folds", 5)
    weights = config.get("weights", None)

    if ensemble_type == "voting":
        return VotingEnsemble(resolved, task_type=task_type, weights=weights)
    elif ensemble_type == "blending":
        return BlendingEnsemble(resolved, task_type=task_type, meta_alpha=meta_alpha)
    elif ensemble_type == "stacking":
        return StackingEnsemble(
            resolved, task_type=task_type, n_folds=n_folds, meta_alpha=meta_alpha
        )
    else:
        raise ValueError(
            f"Unknown ensemble_type: {ensemble_type!r}. "
            f"Supported: 'voting', 'blending', 'stacking'"
        )


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict.

    Nested dicts are merged (not replaced). Non-dict values in overrides
    replace the corresponding base values.

    Examples
    --------
    >>> _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 10}})
    {"a": {"b": 10, "c": 2}}
    """
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def list_models(config_path: str | Path | None = None) -> list[str]:
    """Return sorted list of available model names from the config."""
    return sorted(_load_models_config(config_path).keys())


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=== Available models ===")
    for name in list_models():
        print(f"  {name}")

    print("\n=== Create each model ===")
    for name in list_models():
        model = create_model(name, task_type="regression")
        print(f"\n  {name}:")
        print(f"    type: {type(model).__name__}")
        if hasattr(model, "feature_type"):
            print(f"    features: {model.feature_type}")
            print(f"    fp_radius: {model.fp_radius}, fp_nbits: {model.fp_nbits}")
        if hasattr(model, "get_config"):
            print(f"    config: {model.get_config()}")
        if hasattr(model, "training_config"):
            print(f"    training_config: {model.training_config}")

    print("\n=== Override example ===")
    model = create_model(
        "lgbm_morgan",
        task_type="regression",
        overrides={"lightgbm_params": {"n_estimators": 100, "learning_rate": 0.1}},
    )
    print(f"  Overridden params: {model.get_params()}")

    import IPython
    IPython.embed()
