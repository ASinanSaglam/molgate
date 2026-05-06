"""Dataset bias transforms applied to TDC training splits.

All transforms are pure functions: they return new DataFrames and never
mutate inputs. Bias is always applied to the training split only — val/test
are never touched. This is enforced by the caller in dataset.py.

TDC DataFrames have two relevant columns: 'Drug' (SMILES) and 'Y' (label).
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.utils import resample

logger = logging.getLogger(__name__)

_MIN_MOLECULES_WARNING = 50


# ---------------------------------------------------------------------------
# Bias config models (pydantic v2, discriminated union on 'type')
# ---------------------------------------------------------------------------

class PropertyQuantileBias(BaseModel):
    """Keep only molecules in a quantile band of the Y label distribution.

    Used to study how training on only one end of the property range
    (e.g., only insoluble or only soluble compounds) degrades generalization.
    """

    type: Literal["property_quantile"] = "property_quantile"
    low_quantile: float = 0.0
    high_quantile: float = 1.0

    model_config = {"frozen": True}


class MolWeightRangeBias(BaseModel):
    """Keep only molecules within a molecular weight range.

    Used to study how training on a restricted chemical space
    (fragment-like vs lead-like) affects predictions on the full space.
    """

    type: Literal["mw_range"] = "mw_range"
    min_mw: float | None = None
    max_mw: float | None = None

    model_config = {"frozen": True}


class ClassImbalanceBias(BaseModel):
    """Adjust positive/negative class ratio in the training set.

    Used with classification tasks to study how class imbalance during
    training (common in real screening datasets) affects recall and AUROC.
    Operates only on the majority class; minority is kept intact.
    """

    type: Literal["class_imbalance"] = "class_imbalance"
    positive_fraction: float
    strategy: Literal["undersample_majority", "oversample_minority"] = "undersample_majority"
    seed: int = 42

    model_config = {"frozen": True}


class ScaffoldSubsetBias(BaseModel):
    """Restrict training to molecules with (or without) specific Murcko scaffolds.

    Used to study how training on a narrow chemical series generalizes to
    structurally diverse test sets — directly relevant to scaffold split pessimism.
    """

    type: Literal["scaffold_subset"] = "scaffold_subset"
    scaffold_smiles: list[str]
    invert: bool = False  # if True, exclude these scaffolds instead

    model_config = {"frozen": True}


class ClusterBias(BaseModel):
    """Train on a chemically coherent subset defined by Butina clustering.

    Clusters the training molecules by Tanimoto similarity on Morgan
    fingerprints (ECFP), then keeps only the specified cluster indices.
    Clusters are ordered largest-first (index 0 = most populated cluster).

    This answers a sharper question than MW filtering: if the model has only
    seen chemical series X during training, how well does it generalize to
    structurally distinct series Y? Directly relevant to real screening
    campaigns where training data comes from prior projects.

    Clustering is computed fresh at bias-application time from the training
    DataFrame, so it is fully reproducible from the config parameters alone.

    Args:
        cluster_ids: Indices of clusters to keep for training (0 = largest).
        fingerprint_radius: Morgan fingerprint radius (2 = ECFP4).
        fingerprint_bits: Fingerprint bit vector length.
        butina_cutoff: Tanimoto *distance* threshold (= 1 - similarity).
            0.4 means molecules must share ≥60% Tanimoto similarity to be
            clustered together. Smaller values → more, smaller clusters.
        invert: If True, exclude the listed cluster_ids and keep everything else.
    """

    type: Literal["cluster"] = "cluster"
    cluster_ids: list[int]
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    butina_cutoff: float = 0.4
    invert: bool = False

    model_config = {"frozen": True}


BiasConfig = Annotated[
    PropertyQuantileBias
    | MolWeightRangeBias
    | ClassImbalanceBias
    | ScaffoldSubsetBias
    | ClusterBias,
    Field(discriminator="type"),
]

# Union type for type hints in function signatures (pydantic Annotated doesn't
# work directly in isinstance checks, so we keep a plain union alias too).
AnyBiasConfig = (
    PropertyQuantileBias
    | MolWeightRangeBias
    | ClassImbalanceBias
    | ScaffoldSubsetBias
    | ClusterBias
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_bias(
    df: pd.DataFrame,
    bias: AnyBiasConfig | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply a bias transform to a TDC training DataFrame.

    Args:
        df: TDC-format DataFrame with 'Drug' (SMILES) and 'Y' (label) columns.
        bias: Bias configuration, or None for no transform.
        rng: NumPy random generator (for reproducibility).

    Returns:
        New DataFrame with reset index (input is never mutated).

    Raises:
        ValueError if the bias type is incompatible with the data
        (e.g., ClassImbalanceBias on a regression dataset).
    """
    if bias is None:
        return df.copy()

    if isinstance(bias, PropertyQuantileBias):
        result = _apply_property_quantile(df, bias)
    elif isinstance(bias, MolWeightRangeBias):
        result = _apply_mw_range(df, bias)
    elif isinstance(bias, ClassImbalanceBias):
        result = _apply_class_imbalance(df, bias, rng)
    elif isinstance(bias, ScaffoldSubsetBias):
        result = _apply_scaffold_subset(df, bias)
    elif isinstance(bias, ClusterBias):
        result = _apply_cluster(df, bias)
    else:
        raise ValueError(f"Unknown bias type: {type(bias)}")

    if len(result) < _MIN_MOLECULES_WARNING:
        logger.warning(
            "Bias transform reduced training set to %d molecules "
            "(threshold: %d). Results may be unreliable.",
            len(result),
            _MIN_MOLECULES_WARNING,
        )

    return result.reset_index(drop=True)


def get_cluster_assignments(
    df: pd.DataFrame,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 2048,
    butina_cutoff: float = 0.4,
) -> list[list[int]]:
    """Compute Butina clusters for a DataFrame and return cluster membership.

    Exposed publicly so callers (e.g., notebooks, EDA) can inspect cluster
    sizes and compositions before choosing cluster_ids for a ClusterBias.

    Args:
        df: DataFrame with 'Drug' (SMILES) column.
        fingerprint_radius: Morgan fingerprint radius.
        fingerprint_bits: Fingerprint bit vector length.
        butina_cutoff: Tanimoto distance cutoff.

    Returns:
        List of clusters (sorted largest-first). Each cluster is a list of
        positional indices into df (after reset_index). Molecules that fail
        to parse are assigned to a singleton cluster at the end.
    """
    return _compute_butina_clusters(
        df.reset_index(drop=True),
        fingerprint_radius,
        fingerprint_bits,
        butina_cutoff,
    )


# ---------------------------------------------------------------------------
# Bias implementations
# ---------------------------------------------------------------------------

def _apply_property_quantile(
    df: pd.DataFrame, bias: PropertyQuantileBias
) -> pd.DataFrame:
    y = df["Y"]
    lo = y.quantile(bias.low_quantile)
    hi = y.quantile(bias.high_quantile)
    mask = (y >= lo) & (y <= hi)
    return df[mask].copy()


def _apply_mw_range(df: pd.DataFrame, bias: MolWeightRangeBias) -> pd.DataFrame:
    mws = df["Drug"].apply(_compute_mw)
    mask = pd.Series(True, index=df.index)
    if bias.min_mw is not None:
        mask &= mws >= bias.min_mw
    if bias.max_mw is not None:
        mask &= mws <= bias.max_mw
    return df[mask].copy()


def _apply_class_imbalance(
    df: pd.DataFrame, bias: ClassImbalanceBias, rng: np.random.Generator
) -> pd.DataFrame:
    if not _is_binary(df["Y"]):
        raise ValueError(
            "ClassImbalanceBias requires a binary classification dataset "
            "(Y values must be 0 or 1). Use PropertyQuantileBias for regression."
        )

    pos = df[df["Y"] == 1]
    neg = df[df["Y"] == 0]
    target_frac = bias.positive_fraction
    seed = bias.seed

    if bias.strategy == "undersample_majority":
        n_pos = len(pos)
        n_neg = len(neg)
        if target_frac >= n_pos / (n_pos + n_neg):
            target_n_neg = int(n_pos / target_frac) - n_pos
            neg = neg.sample(n=min(target_n_neg, n_neg), random_state=seed)
        else:
            target_n_pos = int(n_neg * target_frac / (1 - target_frac))
            pos = pos.sample(n=min(target_n_pos, n_pos), random_state=seed)
    elif bias.strategy == "oversample_minority":
        n_total = len(pos) + len(neg)
        target_n_pos = int(n_total * target_frac)
        target_n_neg = n_total - target_n_pos
        pos = resample(pos, n_samples=target_n_pos, replace=True, random_state=seed)
        neg = resample(neg, n_samples=target_n_neg, replace=True, random_state=seed)

    return pd.concat([pos, neg]).sample(frac=1, random_state=seed)


def _apply_scaffold_subset(
    df: pd.DataFrame, bias: ScaffoldSubsetBias
) -> pd.DataFrame:
    target_scaffolds = set(bias.scaffold_smiles)
    mol_scaffolds = df["Drug"].apply(_murcko_scaffold)
    mask = mol_scaffolds.isin(target_scaffolds)
    if bias.invert:
        mask = ~mask
    return df[mask].copy()


def _apply_cluster(df: pd.DataFrame, bias: ClusterBias) -> pd.DataFrame:
    df_reset = df.reset_index(drop=True)
    clusters = _compute_butina_clusters(
        df_reset,
        bias.fingerprint_radius,
        bias.fingerprint_bits,
        bias.butina_cutoff,
    )

    logger.info(
        "Butina clustering (cutoff=%.2f): %d clusters from %d molecules "
        "(largest: %d, selecting ids %s).",
        bias.butina_cutoff,
        len(clusters),
        len(df_reset),
        len(clusters[0]) if clusters else 0,
        bias.cluster_ids,
    )

    rows_to_keep: set[int] = set()
    for cid in bias.cluster_ids:
        if cid < len(clusters):
            rows_to_keep.update(clusters[cid])
        else:
            logger.warning(
                "Cluster id %d requested but only %d clusters exist "
                "(cutoff=%.2f). Skipping.",
                cid, len(clusters), bias.butina_cutoff,
            )

    if bias.invert:
        all_rows = set(range(len(df_reset)))
        rows_to_keep = all_rows - rows_to_keep

    return df_reset.iloc[sorted(rows_to_keep)].copy()


# ---------------------------------------------------------------------------
# Clustering internals
# ---------------------------------------------------------------------------

def _compute_butina_clusters(
    df: pd.DataFrame,
    fingerprint_radius: int,
    fingerprint_bits: int,
    butina_cutoff: float,
) -> list[list[int]]:
    """Run Butina clustering and return clusters as lists of positional indices.

    Butina is O(N²) in the number of molecules for the distance matrix
    computation. For typical TDC training sets (~1000–7000 molecules) this
    takes 1–30 seconds. Molecules that fail to parse are collected into
    singleton clusters appended after the main clusters.

    Returns:
        List of clusters sorted largest-first. Each cluster is a list of
        integer positions into df (0-based, matching a reset index).
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    fps = []
    valid_positions: list[int] = []
    failed_positions: list[int] = []

    for i, smiles in enumerate(df["Drug"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            failed_positions.append(i)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, fingerprint_radius, nBits=fingerprint_bits
        )
        fps.append(fp)
        valid_positions.append(i)

    n = len(fps)
    if n == 0:
        return [[p] for p in failed_positions]

    # Compute upper-triangle Tanimoto distance matrix (required by Butina)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    # Butina returns a tuple of tuples of indices into fps / valid_positions
    raw_clusters = Butina.ClusterData(dists, n, butina_cutoff, isDistData=True)

    # Map fingerprint-space indices back to original DataFrame positions,
    # sort largest-first (Butina already returns sorted, but be explicit)
    clusters: list[list[int]] = [
        [valid_positions[idx] for idx in cluster]
        for cluster in raw_clusters
    ]
    clusters.sort(key=len, reverse=True)

    # Append failed molecules as singletons so they're accounted for
    clusters.extend([p] for p in failed_positions)

    return clusters


# ---------------------------------------------------------------------------
# Molecular utilities
# ---------------------------------------------------------------------------

def _compute_mw(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    return Descriptors.ExactMolWt(mol)


def _murcko_scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def _is_binary(series: pd.Series) -> bool:
    return set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0})
