"""Core prediction logic: SMILES → trained model → prediction.

Load a molgate checkpoint and run inference on one or more SMILES strings.
The public surface is a single function: ``predict_smiles``.

Feature dispatch
----------------
Ensembles need a different feature array per base model
(lgbm_morgan → Morgan FP, lgbm_descriptors → descriptors, gnn → graphs).
The ``ensemble_meta.json`` stored in each ensemble checkpoint encodes this
mapping.  ``_featurize`` reads it and computes each representation only once
even when multiple models share the same feature type.

Invalid SMILES
--------------
SMILES that RDKit cannot parse return ``prediction=NaN, valid=False``.
All other rows in the output DataFrame are guaranteed to have a prediction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def predict_smiles(
    smiles: str | list[str],
    checkpoint_dir: str | Path,
) -> pd.DataFrame:
    """Predict molecular properties for one or more SMILES strings.

    Parameters
    ----------
    smiles : str or list[str]
        One SMILES string or a list of SMILES strings.
    checkpoint_dir : str or Path
        A molgate checkpoint directory (must contain ``manifest.json``).

    Returns
    -------
    pd.DataFrame
        Columns: ``smiles``, ``prediction``, ``valid``.
        Rows with unparseable SMILES have ``prediction=NaN`` and ``valid=False``.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    smiles_list = list(smiles)
    checkpoint_dir = Path(checkpoint_dir)

    from molgate.serve.checkpoint import load_model
    model = load_model(checkpoint_dir)

    # Separate parseable SMILES before any heavy featurization
    from rdkit import Chem
    valid_mask = [Chem.MolFromSmiles(s) is not None for s in smiles_list]
    valid_smiles = [s for s, ok in zip(smiles_list, valid_mask) if ok]

    n_invalid = valid_mask.count(False)
    if n_invalid:
        logger.warning(f"{n_invalid}/{len(smiles_list)} SMILES failed to parse — returning NaN")

    predictions = np.full(len(smiles_list), np.nan)

    if valid_smiles:
        X = _featurize(valid_smiles, checkpoint_dir, model)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            preds = model.predict(X)

        if len(preds) != len(valid_smiles):
            # Graphs can silently drop molecules that fail featurization even
            # after RDKit accepts the SMILES (e.g. exotic atom types).
            logger.warning(
                f"Feature count ({len(preds)}) != valid SMILES count ({len(valid_smiles)}); "
                "some predictions may be misaligned — check for unusual SMILES."
            )

        valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
        for i, pred in zip(valid_indices, preds):
            predictions[i] = float(pred)

    return pd.DataFrame({
        "smiles":     smiles_list,
        "prediction": predictions,
        "valid":      valid_mask,
    })


# ---------------------------------------------------------------------------
# Feature dispatch
# ---------------------------------------------------------------------------

def _featurize(smiles_list: list[str], checkpoint_dir: Path, model: Any) -> Any:
    """Return the right feature representation for the model type.

    Ensembles → X_dict mapping model name → feature array / graph list.
    Single models → bare array or graph list.
    """
    from molgate.models.ensemble import (
        BlendingEnsemble,
        GNNModelWrapper,
        StackingEnsemble,
        VotingEnsemble,
    )

    if isinstance(model, (VotingEnsemble, BlendingEnsemble, StackingEnsemble)):
        meta_path = checkpoint_dir / "ensemble_meta.json"
        with open(meta_path) as f:
            feature_map: dict[str, str] = json.load(f)["feature_map"]
        return _build_X_dict(smiles_list, feature_map)

    if isinstance(model, GNNModelWrapper):
        from molgate.data.featurizer import smiles_list_to_graphs
        return smiles_list_to_graphs(smiles_list)

    # FingerprintModel / SklearnModel
    feature_type = getattr(model, "feature_type", "morgan")
    return _compute_array(smiles_list, feature_type)


def _build_X_dict(smiles_list: list[str], feature_map: dict[str, str]) -> dict[str, Any]:
    """Build ensemble X_dict, computing each feature type only once."""
    from molgate.data.featurizer import (
        compute_descriptors,
        compute_fingerprints,
        smiles_list_to_graphs,
    )

    cache: dict[str, Any] = {}
    needed = set(feature_map.values())

    if "morgan" in needed:
        cache["morgan"] = compute_fingerprints(smiles_list)
    if "descriptors" in needed:
        cache["descriptors"] = compute_descriptors(smiles_list).values
    if "graph" in needed:
        cache["graph"] = smiles_list_to_graphs(smiles_list)

    return {name: cache[ftype] for name, ftype in feature_map.items()}


def _compute_array(smiles_list: list[str], feature_type: str) -> Any:
    from molgate.data.featurizer import compute_descriptors, compute_fingerprints
    if feature_type == "descriptors":
        return compute_descriptors(smiles_list).values
    return compute_fingerprints(smiles_list)
