"""Model serialization and deserialization for molgate.

Checkpoints are self-describing directories — each contains a ``manifest.json``
that records enough metadata for the loader to reconstruct the model without
any external configuration.

Directory layout
----------------
Single model (FingerprintModel / SklearnModel)::

    checkpoints/solubility_lgbm_morgan/
    ├── manifest.json          # model class, task type, feature metadata, metrics
    └── model.joblib           # serialised sklearn-compatible object

GNN model (GNNModelWrapper)::

    checkpoints/solubility_gnn/
    ├── manifest.json          # includes gnn_config dict
    └── model.pt               # {state_dict, gnn_config, task_type}

Ensemble::

    checkpoints/solubility_ensemble_stacking/
    ├── manifest.json          # ensemble_type, base model names + feature map
    ├── meta.joblib            # meta-model (Ridge / LogisticRegression)
    └── base/
        ├── lgbm_morgan.joblib
        ├── lgbm_descriptors.joblib
        └── ...                # one file per base model (gnn → .pt)

The ``feature_map`` in ensemble manifests records which feature type each base
model name expects, e.g. ``{"lgbm_morgan": "morgan", "lgbm_descriptors": "descriptors"}``.
This lets ``serve/predict.py`` compute only the feature types that are actually
needed for a given model.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import molgate

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    checkpoint_dir: str | Path,
    dataset: str = "unknown",
    model_name: str = "unknown",
    metrics: dict[str, float] | None = None,
) -> Path:
    """Serialise a trained model to a checkpoint directory.

    Creates the directory if it does not exist.  Any existing checkpoint
    in the same directory is overwritten.

    Parameters
    ----------
    model : Any
        A fitted FingerprintModel, SklearnModel, GNNModelWrapper, or any
        of the Ensemble classes (VotingEnsemble, BlendingEnsemble,
        StackingEnsemble).
    checkpoint_dir : str | Path
        Directory to write the checkpoint into.
    dataset : str
        Dataset name logged to the manifest (e.g. "solubility").
    model_name : str
        Logical model name logged to the manifest (e.g. "lgbm_morgan").
    metrics : dict[str, float], optional
        Evaluation metrics to store in the manifest for later comparison.

    Returns
    -------
    Path
        Resolved path to the checkpoint directory.
    """
    from molgate.models.ensemble import (
        BlendingEnsemble,
        GNNModelWrapper,
        StackingEnsemble,
        VotingEnsemble,
    )

    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    model_class = type(model).__name__

    if isinstance(model, GNNModelWrapper):
        _save_gnn_wrapper(model, path)
    elif isinstance(model, (VotingEnsemble, BlendingEnsemble, StackingEnsemble)):
        _save_ensemble(model, path)
    else:
        # FingerprintModel, SklearnModel, or any other sklearn-compatible object
        _save_sklearn_compatible(model, path)

    # Write manifest
    manifest = {
        "model_class":       model_class,
        "model_name":        model_name,
        "dataset":           dataset,
        "task_type":         getattr(model, "task_type", "regression"),
        "metrics":           metrics or {},
        "molgate_version":   molgate.__version__,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(path, manifest)

    logger.info(f"Checkpoint saved: {path}")
    return path


def load_model(checkpoint_dir: str | Path) -> Any:
    """Load a model from a checkpoint directory.

    Reads the manifest to determine the model class, then dispatches to the
    appropriate loader.  The returned object is ready to call ``.predict()``.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory containing a ``manifest.json`` and the model file(s).

    Returns
    -------
    model
        Fitted model instance.  For ensembles this is the ensemble object
        with all base models loaded and the meta-model (if any) restored.
    """
    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {path}")

    manifest = _read_manifest(path)
    model_class = manifest["model_class"]

    logger.info(f"Loading checkpoint: {path} (class={model_class})")

    if model_class == "GNNModelWrapper":
        return _load_gnn_wrapper(path, manifest)

    ensemble_classes = {"VotingEnsemble", "BlendingEnsemble", "StackingEnsemble"}
    if model_class in ensemble_classes:
        return _load_ensemble(path, manifest)

    return _load_sklearn_compatible(path, manifest)


def checkpoint_info(checkpoint_dir: str | Path) -> dict:
    """Return the manifest dict for a checkpoint without loading the model."""
    return _read_manifest(Path(checkpoint_dir))


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_sklearn_compatible(model: Any, path: Path) -> None:
    """Save a joblib-serialisable model (FingerprintModel, SklearnModel)."""
    joblib.dump(model, path / "model.joblib")


def _save_gnn_wrapper(wrapper: Any, path: Path) -> None:
    """Save GNNModelWrapper: torch state dict + config."""
    import torch

    if not wrapper._fitted or wrapper.trainer is None:
        raise RuntimeError("GNNModelWrapper must be fitted before saving.")

    gnn = wrapper.gnn_model
    save_dict = {
        "state_dict":      gnn.state_dict(),
        "gnn_config":      gnn.get_config(),
        "task_type":       wrapper.task_type,
        "training_config": wrapper.training_config,
        "val_fraction":    wrapper.val_fraction,
        "seed":            wrapper.seed,
    }
    torch.save(save_dict, path / "model.pt")


def _save_ensemble(ensemble: Any, path: Path) -> None:
    """Save an ensemble: one file per base model + meta-model."""
    from molgate.models.ensemble import GNNModelWrapper

    base_dir = path / "base"
    base_dir.mkdir(exist_ok=True)

    feature_map: dict[str, str] = {}

    for name, model in ensemble.models.items():
        safe_name = name.replace("/", "_")
        if isinstance(model, GNNModelWrapper):
            gnn_subdir = base_dir / safe_name
            gnn_subdir.mkdir(exist_ok=True)
            _save_gnn_wrapper(model, gnn_subdir)
            feature_map[name] = "graph"
        else:
            joblib.dump(model, base_dir / f"{safe_name}.joblib")
            feature_map[name] = getattr(model, "feature_type", "unknown")

    # Meta-model (blending / stacking only)
    if hasattr(ensemble, "meta_model") and ensemble.meta_model is not None:
        joblib.dump(ensemble.meta_model, path / "meta.joblib")

    # Embed ensemble-specific metadata into a sidecar JSON
    ensemble_meta = {
        "ensemble_type": type(ensemble).__name__,
        "feature_map":   feature_map,
        "model_names":   list(ensemble.models.keys()),
        "weights":       getattr(ensemble, "weights", None),
        "meta_alpha":    getattr(ensemble, "meta_alpha", None),
        "n_folds":       getattr(ensemble, "n_folds", None),
    }
    with open(path / "ensemble_meta.json", "w") as f:
        json.dump(ensemble_meta, f, indent=2)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_sklearn_compatible(path: Path, manifest: dict) -> Any:
    model_file = path / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"model.joblib not found in {path}")
    return joblib.load(model_file)


def _load_gnn_wrapper(path: Path, manifest: dict) -> Any:
    import torch
    from molgate.models.ensemble import GNNModelWrapper
    from molgate.models.gnn import MoleculeGNN

    pt_file = path / "model.pt"
    if not pt_file.exists():
        raise FileNotFoundError(f"model.pt not found in {path}")

    save_dict = torch.load(pt_file, map_location="cpu", weights_only=False)
    cfg = save_dict["gnn_config"]

    gnn = MoleculeGNN(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        task_type=cfg["task_type"],
        dropout=cfg["dropout"],
        pool=cfg["pool"],
    )
    gnn.load_state_dict(save_dict["state_dict"])
    gnn.eval()

    wrapper = GNNModelWrapper(
        gnn_model=gnn,
        training_config=save_dict["training_config"],
        task_type=save_dict["task_type"],
        val_fraction=save_dict.get("val_fraction", 0.1),
        seed=save_dict.get("seed", 42),
    )
    # Reconstruct a minimal Trainer so predict() works without re-fitting
    from molgate.training.trainer import Trainer
    trainer = Trainer.from_config(gnn, save_dict["training_config"],
                                  task_type=save_dict["task_type"])
    trainer.model = gnn
    wrapper.trainer = trainer
    wrapper._fitted = True

    return wrapper


def _load_ensemble(path: Path, manifest: dict) -> Any:
    from molgate.models.ensemble import (
        BlendingEnsemble,
        StackingEnsemble,
        VotingEnsemble,
    )

    meta_path = path / "ensemble_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"ensemble_meta.json not found in {path}")

    with open(meta_path) as f:
        emeta = json.load(f)

    base_dir = path / "base"
    models: dict[str, Any] = {}

    for name in emeta["model_names"]:
        safe_name = name.replace("/", "_")
        feature_type = emeta["feature_map"].get(name, "unknown")

        if feature_type == "graph":
            gnn_path = base_dir / safe_name
            sub_manifest = {"model_class": "GNNModelWrapper", "task_type": manifest["task_type"]}
            models[name] = _load_gnn_wrapper(gnn_path, sub_manifest)
        else:
            models[name] = joblib.load(base_dir / f"{safe_name}.joblib")

    task_type = manifest["task_type"]
    etype = emeta["ensemble_type"]

    if etype == "VotingEnsemble":
        ensemble = VotingEnsemble(models, task_type=task_type,
                                  weights=emeta.get("weights"))
    elif etype == "BlendingEnsemble":
        ensemble = BlendingEnsemble(models, task_type=task_type,
                                    meta_alpha=emeta.get("meta_alpha", 1.0))
    elif etype == "StackingEnsemble":
        ensemble = StackingEnsemble(models, task_type=task_type,
                                    n_folds=emeta.get("n_folds", 5),
                                    meta_alpha=emeta.get("meta_alpha", 1.0))
    else:
        raise ValueError(f"Unknown ensemble type in manifest: {etype!r}")

    # Restore meta-model if present (blending / stacking)
    meta_file = path / "meta.joblib"
    if meta_file.exists():
        ensemble.meta_model = joblib.load(meta_file)
        # Restore the meta feature name order for _make_meta_features
        if hasattr(ensemble, "_meta_feature_names"):
            pass  # already set
        else:
            ensemble._meta_feature_names = emeta["model_names"]

    ensemble._fitted = True
    return ensemble


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _write_manifest(path: Path, manifest: dict) -> None:
    with open(path / _MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def _read_manifest(path: Path) -> dict:
    manifest_path = path / _MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json in {path}. "
            "Is this a valid molgate checkpoint directory?"
        )
    with open(manifest_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Convenience: save the best model from an ensemble comparison table
# ---------------------------------------------------------------------------

def save_best_from_results(
    results_df: "pd.DataFrame",
    ensembles: dict[str, Any],
    base_models: dict[str, Any],
    checkpoint_root: str | Path,
    dataset: str,
    metric_col: str = "test_mae",
) -> Path:
    """Save the best model from a tune_and_compare() results DataFrame.

    Looks up the winning model name in ``results_df`` (lowest ``metric_col``),
    finds it in ``ensembles`` or ``base_models``, and saves it.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``tune_and_compare()`` or a similar comparison table.
    ensembles : dict[str, ensemble]
        Fitted ensemble objects keyed by name.
    base_models : dict[str, model]
        Fitted base model objects keyed by name.
    checkpoint_root : str | Path
        Root directory under which checkpoints are stored.
    dataset : str
        Dataset name (e.g. "solubility").
    metric_col : str
        Column used to pick the winner (lower is better).

    Returns
    -------
    Path
        Checkpoint directory of the saved winner.
    """
    winner_name = results_df.sort_values(metric_col).iloc[0]["name"]
    winner_metrics = results_df.set_index("name").loc[winner_name].to_dict()

    model = ensembles.get(winner_name) or base_models.get(winner_name)
    if model is None:
        raise KeyError(
            f"Winner '{winner_name}' not found in ensembles or base_models dicts."
        )

    checkpoint_dir = Path(checkpoint_root) / dataset / winner_name
    save_model(
        model,
        checkpoint_dir,
        dataset=dataset,
        model_name=winner_name,
        metrics={k: v for k, v in winner_metrics.items()
                 if isinstance(v, (int, float)) and not isinstance(v, bool)},
    )
    logger.info(f"Best model '{winner_name}' saved to {checkpoint_dir}")
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.featurizer import compute_descriptors, compute_fingerprints
    from molgate.data.loaders import load_tdc_benchmark_split
    from molgate.data.splits import random_split
    from molgate.models.factory import create_model
    from molgate.training.metrics import compute_metrics

    print("Loading solubility TDC benchmark split...")
    train_val_df, test_df = load_tdc_benchmark_split("solubility")
    sub = random_split(train_val_df, val_frac=0.1, test_frac=0.0, seed=42)
    train_df, val_df = sub["train"], sub["val"]

    def featurize(df):
        sm = df["smiles"].tolist()
        fp   = compute_fingerprints(sm)
        desc = compute_descriptors(sm).values
        return {"lgbm_morgan": fp, "lgbm_descriptors": desc,
                "rf_morgan": fp, "ridge_descriptors": desc}

    X_train, X_val, X_test = featurize(train_df), featurize(val_df), featurize(test_df)
    y_train, y_val, y_test = train_df["y"].values, val_df["y"].values, test_df["y"].values

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Save / load FingerprintModel ---
        print("\n--- FingerprintModel round-trip ---")
        m = create_model("lgbm_morgan", task_type="regression")
        m.fit(X_train["lgbm_morgan"], y_train)
        preds_before = m.predict(X_test["lgbm_morgan"])

        ckpt = save_model(m, f"{tmpdir}/lgbm_morgan",
                          dataset="solubility", model_name="lgbm_morgan",
                          metrics=compute_metrics(y_test, preds_before, "regression"))
        m2 = load_model(ckpt)
        preds_after = m2.predict(X_test["lgbm_morgan"])
        assert np.allclose(preds_before, preds_after), "FingerprintModel round-trip FAILED"
        print(f"  MAE before={compute_metrics(y_test, preds_before, 'regression')['mae']:.4f} "
              f"after={compute_metrics(y_test, preds_after, 'regression')['mae']:.4f}  ✓")

        # --- Save / load VotingEnsemble ---
        print("\n--- VotingEnsemble round-trip ---")
        from molgate.models.ensemble import VotingEnsemble
        ev = create_model("ensemble_voting_fp", task_type="regression")
        ev.fit(X_train, y_train)
        preds_before = ev.predict(X_test)

        ckpt = save_model(ev, f"{tmpdir}/ensemble_voting",
                          dataset="solubility", model_name="ensemble_voting_fp",
                          metrics=compute_metrics(y_test, preds_before, "regression"))
        ev2 = load_model(ckpt)
        preds_after = ev2.predict(X_test)
        assert np.allclose(preds_before, preds_after), "VotingEnsemble round-trip FAILED"
        print(f"  MAE before={compute_metrics(y_test, preds_before, 'regression')['mae']:.4f} "
              f"after={compute_metrics(y_test, preds_after, 'regression')['mae']:.4f}  ✓")

        print("\nAll round-trip tests passed.")
        info = checkpoint_info(ckpt)
        print(f"Manifest: {json.dumps(info, indent=2)}")
