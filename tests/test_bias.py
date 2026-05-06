"""Tests for admet/analysis/bias.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from admet.analysis.bias import (
    ClassImbalanceBias,
    ClusterBias,
    MolWeightRangeBias,
    PropertyQuantileBias,
    ScaffoldSubsetBias,
    apply_bias,
    get_cluster_assignments,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestPropertyQuantileBias:
    def test_bottom_third(self, regression_df, rng):
        bias = PropertyQuantileBias(low_quantile=0.0, high_quantile=0.33)
        result = apply_bias(regression_df, bias, rng)
        assert len(result) <= len(regression_df)
        assert result["Y"].max() <= regression_df["Y"].quantile(0.33) + 1e-9

    def test_top_third(self, regression_df, rng):
        bias = PropertyQuantileBias(low_quantile=0.67, high_quantile=1.0)
        result = apply_bias(regression_df, bias, rng)
        assert result["Y"].min() >= regression_df["Y"].quantile(0.67) - 1e-9

    def test_no_mutation(self, regression_df, rng):
        original_len = len(regression_df)
        bias = PropertyQuantileBias(low_quantile=0.0, high_quantile=0.5)
        apply_bias(regression_df, bias, rng)
        assert len(regression_df) == original_len


class TestMolWeightRangeBias:
    def test_max_mw_filter(self, regression_df, rng):
        bias = MolWeightRangeBias(max_mw=200.0)
        result = apply_bias(regression_df, bias, rng)
        assert len(result) > 0

    def test_min_mw_filter(self, regression_df, rng):
        bias = MolWeightRangeBias(min_mw=150.0)
        result = apply_bias(regression_df, bias, rng)
        assert len(result) > 0

    def test_no_filter(self, regression_df, rng):
        bias = MolWeightRangeBias()
        result = apply_bias(regression_df, bias, rng)
        assert len(result) == len(regression_df)

    def test_combined_range(self, regression_df, rng):
        bias = MolWeightRangeBias(min_mw=50.0, max_mw=500.0)
        result = apply_bias(regression_df, bias, rng)
        assert len(result) <= len(regression_df)


class TestClassImbalanceBias:
    def test_undersample_to_balanced(self, classification_df, rng):
        bias = ClassImbalanceBias(positive_fraction=0.5, strategy="undersample_majority")
        result = apply_bias(classification_df, bias, rng)
        pos = (result["Y"] == 1).sum()
        neg = (result["Y"] == 0).sum()
        assert pos + neg == len(result)
        assert abs(pos - neg) <= 2

    def test_raises_on_regression_data(self, regression_df, rng):
        bias = ClassImbalanceBias(positive_fraction=0.5, strategy="undersample_majority")
        with pytest.raises(ValueError, match="binary classification"):
            apply_bias(regression_df, bias, rng)


class TestClusterBias:
    def test_keeps_subset_of_molecules(self, regression_df, rng):
        bias = ClusterBias(cluster_ids=[0], butina_cutoff=0.4)
        result = apply_bias(regression_df, bias, rng)
        assert 0 < len(result) <= len(regression_df)

    def test_invert_is_complement(self, regression_df, rng):
        bias_keep = ClusterBias(cluster_ids=[0], butina_cutoff=0.4)
        bias_exclude = ClusterBias(cluster_ids=[0], butina_cutoff=0.4, invert=True)
        keep = apply_bias(regression_df, bias_keep, rng)
        exclude = apply_bias(regression_df, bias_exclude, rng)
        # Keep + exclude should cover all valid molecules (may differ from full df
        # only if some SMILES fail to parse, but our fixture SMILES are all valid)
        assert len(keep) + len(exclude) == len(regression_df)

    def test_multiple_clusters_larger_than_single(self, regression_df, rng):
        bias_one = ClusterBias(cluster_ids=[0], butina_cutoff=0.4)
        bias_three = ClusterBias(cluster_ids=[0, 1, 2], butina_cutoff=0.4)
        one = apply_bias(regression_df, bias_one, rng)
        three = apply_bias(regression_df, bias_three, rng)
        assert len(three) >= len(one)

    def test_out_of_range_cluster_id_warns(self, regression_df, rng, caplog):
        import logging
        bias = ClusterBias(cluster_ids=[999], butina_cutoff=0.4)
        with caplog.at_level(logging.WARNING):
            result = apply_bias(regression_df, bias, rng)
        assert any("999" in r.message for r in caplog.records)

    def test_index_is_reset(self, regression_df, rng):
        bias = ClusterBias(cluster_ids=[0], butina_cutoff=0.4)
        result = apply_bias(regression_df, bias, rng)
        assert list(result.index) == list(range(len(result)))

    def test_output_is_not_input(self, regression_df, rng):
        bias = ClusterBias(cluster_ids=[0], butina_cutoff=0.4)
        result = apply_bias(regression_df, bias, rng)
        assert result is not regression_df

    def test_tight_cutoff_produces_more_clusters(self, regression_df, rng):
        """Tighter Tanimoto distance cutoff → more, smaller clusters."""
        clusters_loose = get_cluster_assignments(regression_df, butina_cutoff=0.8)
        clusters_tight = get_cluster_assignments(regression_df, butina_cutoff=0.1)
        assert len(clusters_tight) >= len(clusters_loose)

    def test_get_cluster_assignments_covers_all_molecules(self, regression_df, rng):
        clusters = get_cluster_assignments(regression_df, butina_cutoff=0.4)
        total_assigned = sum(len(c) for c in clusters)
        assert total_assigned == len(regression_df)


class TestNoBias:
    def test_none_returns_copy(self, regression_df, rng):
        result = apply_bias(regression_df, None, rng)
        assert len(result) == len(regression_df)
        assert result is not regression_df


class TestApplyBias:
    def test_output_is_not_input(self, regression_df, rng):
        bias = PropertyQuantileBias(low_quantile=0.0, high_quantile=1.0)
        result = apply_bias(regression_df, bias, rng)
        assert result is not regression_df

    def test_index_is_reset(self, regression_df, rng):
        bias = PropertyQuantileBias(low_quantile=0.0, high_quantile=0.5)
        result = apply_bias(regression_df, bias, rng)
        assert list(result.index) == list(range(len(result)))
