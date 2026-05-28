"""Tests for analysis/descriptors.py."""

import numpy as np
import pandas as pd
import pytest

from molgate.analysis.descriptors import (
    compare_splits,
    descriptor_analysis,
    descriptor_summary,
    descriptor_target_correlations,
)
from molgate.data.featurizer import compute_descriptors


class TestDescriptorSummary:
    """Tests for descriptor_summary."""

    def test_returns_dataframe(self, sample_smiles):
        desc = compute_descriptors(sample_smiles)
        result = descriptor_summary(desc)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_smiles):
        desc = compute_descriptors(sample_smiles)
        result = descriptor_summary(desc)
        expected_cols = {"mean", "std", "min", "q25", "median", "q75", "max", "skewness", "kurtosis"}
        assert expected_cols == set(result.columns)

    def test_one_row_per_descriptor(self, sample_smiles):
        desc = compute_descriptors(sample_smiles)
        result = descriptor_summary(desc)
        assert len(result) == len(desc.columns)

    def test_mean_in_range(self, sample_smiles):
        """Mean should be between min and max for every descriptor."""
        desc = compute_descriptors(sample_smiles)
        result = descriptor_summary(desc)
        for _, row in result.iterrows():
            assert row["min"] <= row["mean"] <= row["max"]


class TestCompareSplits:
    """Tests for compare_splits (KS test)."""

    def test_identical_splits_low_ks(self, sample_smiles):
        """KS stat should be 0 when comparing a dataset to itself."""
        desc = compute_descriptors(sample_smiles)
        result = compare_splits(desc, desc)
        assert (result["ks_statistic"] == 0.0).all()

    def test_returns_expected_columns(self, sample_smiles):
        desc = compute_descriptors(sample_smiles)
        result = compare_splits(desc, desc, label_a="a", label_b="b")
        assert "ks_statistic" in result.columns
        assert "p_value" in result.columns
        assert "significant" in result.columns
        assert "mean_a" in result.columns
        assert "mean_b" in result.columns

    def test_sorted_by_ks(self, sample_smiles):
        """Results should be sorted by KS statistic descending."""
        desc_a = compute_descriptors(sample_smiles[:5])
        desc_b = compute_descriptors(sample_smiles[5:])
        result = compare_splits(desc_a, desc_b)
        ks_vals = result["ks_statistic"].values
        assert all(ks_vals[i] >= ks_vals[i + 1] for i in range(len(ks_vals) - 1))

    def test_mismatched_columns_raises(self, sample_smiles):
        desc = compute_descriptors(sample_smiles)
        desc_partial = desc.drop(columns=["mw"])
        with pytest.raises(ValueError, match="columns don't match"):
            compare_splits(desc, desc_partial)


class TestDescriptorTargetCorrelations:
    """Tests for descriptor_target_correlations."""

    def test_returns_dataframe(self, sample_smiles, sample_targets):
        desc = compute_descriptors(sample_smiles)
        result = descriptor_target_correlations(desc, sample_targets)
        assert isinstance(result, pd.DataFrame)

    def test_has_correlation_columns(self, sample_smiles, sample_targets):
        desc = compute_descriptors(sample_smiles)
        result = descriptor_target_correlations(desc, sample_targets)
        for col in ["pearson_r", "spearman_r", "abs_pearson", "abs_spearman"]:
            assert col in result.columns

    def test_sorted_by_abs_spearman(self, sample_smiles, sample_targets):
        desc = compute_descriptors(sample_smiles)
        result = descriptor_target_correlations(desc, sample_targets)
        vals = result["abs_spearman"].values
        # Check descending (allow NaN at end)
        valid = vals[~np.isnan(vals)]
        assert all(valid[i] >= valid[i + 1] for i in range(len(valid) - 1))

    def test_length_mismatch_raises(self, sample_smiles):
        desc = compute_descriptors(sample_smiles)
        with pytest.raises(ValueError, match="Length mismatch"):
            descriptor_target_correlations(desc, np.array([1.0, 2.0]))

    def test_correlations_bounded(self, sample_smiles, sample_targets):
        """Pearson and Spearman r should be in [-1, 1]."""
        desc = compute_descriptors(sample_smiles)
        result = descriptor_target_correlations(desc, sample_targets)
        for col in ["pearson_r", "spearman_r"]:
            vals = result[col].dropna()
            assert (vals >= -1.0).all() and (vals <= 1.0).all()


class TestDescriptorAnalysis:
    """Tests for the convenience descriptor_analysis function."""

    def test_summary_only(self, sample_smiles):
        report = descriptor_analysis(sample_smiles)
        assert "summary" in report
        assert "ks_comparison" not in report
        assert "correlations" not in report

    def test_with_comparison(self, sample_smiles):
        report = descriptor_analysis(
            sample_smiles[:5],
            smiles_b=sample_smiles[5:],
        )
        assert "ks_comparison" in report
        assert "n_significant_ks" in report

    def test_with_target(self, sample_smiles, sample_targets):
        report = descriptor_analysis(
            sample_smiles,
            target_a=sample_targets,
        )
        assert "correlations" in report
        assert "top_correlated_descriptor" in report
