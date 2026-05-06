"""TDC dataset loading, split dispatch, and PyG Dataset wrapping.

Two loading modes:
- load_benchmark_split(): uses admet_group (fixed test set, leaderboard-compatible).
  All bias experiments should use this so results are directly comparable to
  the TDC leaderboard. Default for all 9 pipeline properties.
- load_tdc_split(): uses single_pred with dynamic splits. Kept for datasets
  outside the benchmark group and exploration use cases.

Bias transforms are always applied to the training split only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data, Dataset

from admet.analysis.bias import AnyBiasConfig, apply_bias
from admet.featurizer import MolFeaturizer

logger = logging.getLogger(__name__)

TaskType = Literal["regression", "classification"]
SplitMethod = Literal["random", "scaffold"]

# Datasets with known split support. Requesting an unlisted method raises
# ValueError with a clear message rather than a cryptic TDC error.
SUPPORTED_SPLITS: dict[str, list[SplitMethod]] = {
    "solubility_aqsoldb": ["random", "scaffold"],
    "herg": ["random", "scaffold"],
    "dili": ["random", "scaffold"],
    "caco2_wang": ["random", "scaffold"],
    "cyp3a4_substrate_carbonmangels": ["random", "scaffold"],
    "cyp2d6_substrate_carbonmangels": ["random", "scaffold"],
    "cyp2c9_substrate_carbonmangels": ["random", "scaffold"],
    "half_life_obach": ["random", "scaffold"],
    "clearance_hepatocyte_az": ["random", "scaffold"],
}

# Task type for every dataset in the admet_group benchmark (22 datasets).
# Used by benchmark_eval.py to configure models without a properties.yaml lookup.
BENCHMARK_TASK_TYPES: dict[str, TaskType] = {
    "caco2_wang": "regression",
    "hia_hou": "classification",
    "pgp_broccatelli": "classification",
    "bioavailability_ma": "classification",
    "lipophilicity_astrazeneca": "regression",
    "solubility_aqsoldb": "regression",
    "bbb_martins": "classification",
    "ppbr_az": "regression",
    "vdss_lombardo": "regression",
    "cyp2d6_veith": "classification",
    "cyp3a4_veith": "classification",
    "cyp2c9_veith": "classification",
    "cyp2d6_substrate_carbonmangels": "classification",
    "cyp3a4_substrate_carbonmangels": "classification",
    "cyp2c9_substrate_carbonmangels": "classification",
    "half_life_obach": "regression",
    "clearance_microsome_az": "regression",
    "clearance_hepatocyte_az": "regression",
    "herg": "classification",
    "ames": "classification",
    "dili": "classification",
    "ld50_zhu": "regression",
}


class ADMETDataset(Dataset):
    """PyG Dataset wrapping a single TDC split.

    Featurizes all molecules on construction (eager, in-memory).
    Invalid SMILES are skipped with a warning; the compacted data list is
    stored alongside a positional index so predictions can be re-aligned
    with the original DataFrame for leaderboard submission.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        featurizer: MolFeaturizer,
        task_type: TaskType,
    ) -> None:
        """
        Args:
            df: TDC-format DataFrame with 'Drug' (SMILES) and 'Y' columns.
            featurizer: Molecular featurizer producing PyG Data objects.
            task_type: "regression" or "classification".
        """
        super().__init__()
        self._task_type = task_type
        self._df = df.reset_index(drop=True)  # retained for statistics and alignment
        self._data_list: list[Data] = []
        # Maps position in _data_list → row index in _df (needed for alignment)
        self._valid_positions: list[int] = []
        self._n_total = len(df)

        n_failed = 0
        for pos, row in self._df.iterrows():
            smiles = row["Drug"]
            label = float(row["Y"])
            data = featurizer.featurize(smiles)
            if data is None:
                n_failed += 1
                continue
            data.y = torch.tensor([[label]], dtype=torch.float)
            self._data_list.append(data)
            self._valid_positions.append(int(pos))

        if n_failed > 0:
            logger.warning(
                "%d / %d SMILES failed featurization and were skipped.",
                n_failed, len(df),
            )

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    @property
    def num_node_features(self) -> int:
        if not self._data_list:
            return 0
        return self._data_list[0].x.shape[1]

    @property
    def num_edge_features(self) -> int:
        if not self._data_list:
            return 0
        return self._data_list[0].edge_attr.shape[1]

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def n_failed(self) -> int:
        return self._n_total - len(self._data_list)


# ---------------------------------------------------------------------------
# Benchmark-mode loading (leaderboard-compatible, fixed test set)
# ---------------------------------------------------------------------------

def load_benchmark_split(
    benchmark_name: str,
    split_type: SplitMethod,
    seed: int,
    data_dir: Path,
    featurizer: MolFeaturizer,
    task_type: TaskType,
    train_bias: AnyBiasConfig | None = None,
) -> tuple[ADMETDataset, ADMETDataset, ADMETDataset]:
    """Load using admet_group for leaderboard-compatible evaluation.

    The test set is the fixed TDC benchmark test set — identical across all
    seeds and bias variants, so all experiments evaluate on the same molecules
    and are directly comparable to the leaderboard.

    Train and val are derived from the fixed train_val pool using the
    specified split_type (scaffold or random). Bias is applied to the
    training split only; val and test are never modified.

    Args:
        benchmark_name: TDC benchmark dataset name (e.g. "solubility_aqsoldb").
        split_type: How to partition train_val → train/val ("scaffold" or "random").
        seed: Random seed for the train/val partition.
        data_dir: Directory for TDC data cache.
        featurizer: Molecular featurizer.
        task_type: "regression" or "classification".
        train_bias: Optional bias applied to train split only.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    from tdc.benchmark_group import admet_group

    data_dir.mkdir(parents=True, exist_ok=True)
    bg = admet_group(path=str(data_dir))

    # Fixed test set — always the same regardless of seed or bias
    benchmark_data = bg.get(benchmark_name)
    test_df: pd.DataFrame = benchmark_data["test"]

    # Train / val from the fixed train_val pool
    train_df, val_df = bg.get_train_valid_split(
        benchmark=benchmark_name,
        split_type=split_type,
        seed=seed,
    )

    if train_bias is not None:
        rng = np.random.default_rng(seed)
        original_size = len(train_df)
        train_df = apply_bias(train_df, train_bias, rng)
        logger.info(
            "Bias applied: %s → training set %d → %d molecules.",
            type(train_bias).__name__, original_size, len(train_df),
        )

    train_dataset = ADMETDataset(train_df, featurizer, task_type)
    val_dataset = ADMETDataset(val_df, featurizer, task_type)
    test_dataset = ADMETDataset(test_df, featurizer, task_type)

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Exploration-mode loading (single_pred, dynamic splits)
# ---------------------------------------------------------------------------

def load_tdc_split(
    tdc_dataset_name: str,
    split_method: SplitMethod,
    seed: int,
    data_dir: Path,
    featurizer: MolFeaturizer,
    task_type: TaskType,
    train_bias: AnyBiasConfig | None = None,
) -> tuple[ADMETDataset, ADMETDataset, ADMETDataset]:
    """Load via single_pred with dynamic splits.

    Use this for datasets outside the admet_group benchmark, or when you
    need full control over split composition. Results are NOT directly
    comparable to the TDC leaderboard because the test set varies by seed.

    Args:
        tdc_dataset_name: TDC single_pred dataset name.
        split_method: "random" or "scaffold".
        seed: Seed for TDC's get_split().
        data_dir: TDC cache directory.
        featurizer: Molecular featurizer.
        task_type: "regression" or "classification".
        train_bias: Optional bias transform for the training split.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    _validate_split_method(tdc_dataset_name, split_method)

    splits = _fetch_tdc_splits(tdc_dataset_name, split_method, seed, data_dir)
    train_df: pd.DataFrame = splits["train"]
    val_df: pd.DataFrame = splits["valid"]
    test_df: pd.DataFrame = splits["test"]

    if train_bias is not None:
        rng = np.random.default_rng(seed)
        original_size = len(train_df)
        train_df = apply_bias(train_df, train_bias, rng)
        logger.info(
            "Bias applied: %s → training set %d → %d molecules.",
            type(train_bias).__name__, original_size, len(train_df),
        )

    train_dataset = ADMETDataset(train_df, featurizer, task_type)
    val_dataset = ADMETDataset(val_df, featurizer, task_type)
    test_dataset = ADMETDataset(test_df, featurizer, task_type)

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_split_statistics(df: pd.DataFrame, task_type: TaskType) -> dict[str, float]:
    """Compute summary statistics for a TDC split DataFrame.

    Used by train.py and runner.py to log dataset metadata to W&B and
    result JSONs. Call once per split on the raw DataFrame before featurization.

    Returns a flat dict of floats with keys: n, mw_mean, mw_std, mw_median,
    mw_min, mw_max, y_mean, y_std, y_min, y_max, pos_fraction (classification).
    """
    stats: dict[str, float] = {"n": float(len(df))}

    mws = df["Drug"].apply(_safe_mw).dropna()
    if len(mws) > 0:
        stats["mw_mean"] = float(mws.mean())
        stats["mw_std"] = float(mws.std())
        stats["mw_median"] = float(mws.median())
        stats["mw_min"] = float(mws.min())
        stats["mw_max"] = float(mws.max())

    y = df["Y"].dropna()
    stats["y_mean"] = float(y.mean())
    stats["y_std"] = float(y.std())
    stats["y_min"] = float(y.min())
    stats["y_max"] = float(y.max())

    if task_type == "classification":
        stats["pos_fraction"] = float((y == 1).sum() / len(y))

    return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_split_method(dataset_name: str, split_method: SplitMethod) -> None:
    supported = SUPPORTED_SPLITS.get(dataset_name)
    if supported is None:
        logger.warning(
            "Dataset '%s' not in SUPPORTED_SPLITS — proceeding with '%s' split "
            "but support is unverified.",
            dataset_name, split_method,
        )
        return
    if split_method not in supported:
        raise ValueError(
            f"Split method '{split_method}' is not supported for dataset "
            f"'{dataset_name}'. Supported methods: {supported}"
        )


def _fetch_tdc_splits(
    dataset_name: str,
    split_method: str,
    seed: int,
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    from tdc.single_pred import ADME, Tox

    data_dir.mkdir(parents=True, exist_ok=True)
    path = str(data_dir)

    try:
        data = ADME(name=dataset_name, path=path)
    except Exception:
        try:
            data = Tox(name=dataset_name, path=path)
        except Exception as exc:
            raise ValueError(
                f"Could not load TDC dataset '{dataset_name}'. "
                "Check the name against TDC's ADME/Tox benchmarks."
            ) from exc

    return data.get_split(method=split_method, seed=seed)


def _safe_mw(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.ExactMolWt(mol)
