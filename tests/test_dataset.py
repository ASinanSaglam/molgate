"""Tests for admet/dataset.py (unit tests that don't require TDC download)."""

from __future__ import annotations

import pandas as pd
import pytest


def test_admet_dataset_basic(regression_df, rdkit_featurizer):
    from admet.dataset import ADMETDataset
    ds = ADMETDataset(regression_df, rdkit_featurizer, "regression")
    assert len(ds) > 0
    assert ds.num_node_features == 30
    assert ds.num_edge_features == 11
    assert ds.task_type == "regression"


def test_admet_dataset_y_shape(regression_df, rdkit_featurizer):
    from admet.dataset import ADMETDataset
    ds = ADMETDataset(regression_df, rdkit_featurizer, "regression")
    data = ds[0]
    assert data.y.shape == (1, 1)


def test_admet_dataset_skips_invalid_smiles(rdkit_featurizer):
    from admet.dataset import ADMETDataset
    df = pd.DataFrame({"Drug": ["CCO", "INVALID!!!", "c1ccccc1"], "Y": [1.0, 2.0, 3.0]})
    ds = ADMETDataset(df, rdkit_featurizer, "regression")
    assert len(ds) == 2  # one skipped


def test_admet_dataset_stores_raw_df(regression_df, rdkit_featurizer):
    from admet.dataset import ADMETDataset
    ds = ADMETDataset(regression_df, rdkit_featurizer, "regression")
    assert hasattr(ds, "_df")
    assert len(ds._df) == len(regression_df)


def test_compute_split_statistics_regression(regression_df):
    from admet.dataset import compute_split_statistics
    stats = compute_split_statistics(regression_df, "regression")
    assert "n" in stats
    assert "mw_mean" in stats
    assert "y_mean" in stats
    assert "pos_fraction" not in stats
    assert stats["n"] == len(regression_df)


def test_compute_split_statistics_classification(classification_df):
    from admet.dataset import compute_split_statistics
    stats = compute_split_statistics(classification_df, "classification")
    assert "pos_fraction" in stats
    assert 0.0 <= stats["pos_fraction"] <= 1.0


def test_validate_split_method_raises():
    from admet.dataset import _validate_split_method
    with pytest.raises(ValueError, match="temporal"):
        _validate_split_method("solubility_aqsoldb", "temporal")


def test_validate_split_method_ok():
    from admet.dataset import _validate_split_method
    _validate_split_method("solubility_aqsoldb", "scaffold")  # should not raise


def test_predict_aligned_length(regression_df, rdkit_featurizer):
    """predict_aligned must return one value per row in the original DataFrame."""
    import torch
    from torch_geometric.data import Batch
    from admet.dataset import ADMETDataset
    from admet.evaluate import predict_aligned
    from admet.model import ADMETModel, ModelConfig

    # Inject one invalid SMILES so a featurization failure occurs
    df_with_bad = regression_df.copy()
    df_with_bad.loc[0, "Drug"] = "INVALID!!!"

    ds = ADMETDataset(df_with_bad, rdkit_featurizer, "regression")
    assert ds.n_failed == 1
    assert len(ds) == len(df_with_bad) - 1

    model = ADMETModel(ModelConfig(task_type="regression"))
    aligned = predict_aligned(model, ds)

    assert aligned.shape == (len(df_with_bad),)  # one per original row
    # Position 0 (failed) should be filled with the fallback mean
    assert aligned[0] == pytest.approx(aligned[ds._valid_positions].mean(), abs=1e-5)
