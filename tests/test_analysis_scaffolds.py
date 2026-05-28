"""Tests for analysis/scaffolds.py."""

import pytest

from molgate.analysis.scaffolds import (
    get_scaffold,
    get_scaffolds,
    scaffold_analysis,
    scaffold_diversity,
    scaffold_frequency_table,
    scaffold_overlap,
    scaffold_target_stats,
)


class TestGetScaffold:
    """Tests for single-molecule scaffold extraction."""

    def test_benzene_derivative(self):
        """Aspirin's scaffold is benzene."""
        result = get_scaffold("CC(=O)Oc1ccccc1C(=O)O")
        assert result == "c1ccccc1"

    def test_acyclic_returns_none(self):
        """Molecules with no rings should return None."""
        assert get_scaffold("CCO") is None
        assert get_scaffold("CC(=O)O") is None

    def test_invalid_smiles_returns_none(self):
        assert get_scaffold("not_a_molecule") is None

    def test_biphenyl(self):
        """Biphenyl scaffold should be biphenyl itself (two connected rings)."""
        result = get_scaffold("c1ccc(-c2ccccc2)cc1")
        assert result is not None
        assert result == "c1ccc(-c2ccccc2)cc1"


class TestGetScaffolds:
    """Tests for batch scaffold extraction."""

    def test_returns_correct_length(self, sample_smiles):
        result = get_scaffolds(sample_smiles)
        assert len(result) == len(sample_smiles)

    def test_acyclic_are_none(self):
        result = get_scaffolds(["CCO", "CC(=O)O", "c1ccccc1"])
        assert result[0] is None   # ethanol
        assert result[1] is None   # acetic acid
        assert result[2] is not None  # benzene


class TestScaffoldDiversity:
    """Tests for scaffold diversity metrics."""

    def test_returns_expected_keys(self, sample_smiles):
        result = scaffold_diversity(sample_smiles)
        expected_keys = {
            "n_molecules", "n_scaffolds", "n_acyclic", "scaffold_ratio",
            "singleton_fraction", "top_1_coverage", "top_5_coverage",
            "top_10_coverage", "top_20_coverage", "scaffold_counts",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_n_molecules_correct(self, sample_smiles):
        result = scaffold_diversity(sample_smiles)
        assert result["n_molecules"] == len(sample_smiles)

    def test_scaffold_ratio_bounded(self, sample_smiles):
        result = scaffold_diversity(sample_smiles)
        assert 0.0 <= result["scaffold_ratio"] <= 1.0

    def test_singleton_fraction_bounded(self, sample_smiles):
        result = scaffold_diversity(sample_smiles)
        assert 0.0 <= result["singleton_fraction"] <= 1.0

    def test_all_same_scaffold(self):
        """All benzene derivatives → 1 unique scaffold, ratio near 0."""
        smiles = ["c1ccccc1", "Oc1ccccc1", "Cc1ccccc1", "Nc1ccccc1"]
        result = scaffold_diversity(smiles)
        assert result["n_scaffolds"] == 1
        assert result["singleton_fraction"] == 0.0


class TestScaffoldFrequencyTable:
    """Tests for scaffold_frequency_table."""

    def test_returns_dataframe(self, sample_smiles):
        result = scaffold_frequency_table(sample_smiles)
        assert "scaffold" in result.columns
        assert "count" in result.columns
        assert "fraction" in result.columns
        assert "cumulative_fraction" in result.columns

    def test_sorted_descending(self, sample_smiles):
        result = scaffold_frequency_table(sample_smiles)
        counts = result["count"].values
        assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))

    def test_top_n_limits_rows(self, sample_smiles):
        result = scaffold_frequency_table(sample_smiles, top_n=3)
        assert len(result) <= 3


class TestScaffoldOverlap:
    """Tests for scaffold overlap between splits."""

    def test_identical_sets_jaccard_one(self, sample_smiles):
        """Same SMILES in both → Jaccard = 1."""
        result = scaffold_overlap(sample_smiles, sample_smiles)
        assert result["jaccard"] == pytest.approx(1.0)

    def test_disjoint_sets_jaccard_zero(self):
        """Completely different scaffolds → Jaccard = 0."""
        set_a = ["c1ccccc1", "Oc1ccccc1"]        # benzene scaffold
        set_b = ["c1ccncc1", "Cc1ccncc1"]         # pyridine scaffold
        result = scaffold_overlap(set_a, set_b)
        assert result["jaccard"] == pytest.approx(0.0)

    def test_overlap_keys(self, sample_smiles):
        result = scaffold_overlap(
            sample_smiles[:5], sample_smiles[5:],
            label_a="train", label_b="test",
        )
        assert "n_shared" in result
        assert "jaccard" in result
        assert "frac_train_in_shared" in result
        assert "frac_test_in_shared" in result


class TestScaffoldTargetStats:
    """Tests for per-scaffold target statistics."""

    def test_returns_dataframe(self, sample_smiles, sample_targets):
        result = scaffold_target_stats(sample_smiles, sample_targets)
        assert "scaffold" in result.columns
        assert "target_mean" in result.columns
        assert "count" in result.columns

    def test_respects_top_n(self, sample_smiles, sample_targets):
        result = scaffold_target_stats(sample_smiles, sample_targets, top_n=2)
        assert len(result) <= 2


class TestScaffoldAnalysis:
    """Tests for the convenience scaffold_analysis function."""

    def test_basic_report(self, sample_smiles):
        report = scaffold_analysis(sample_smiles)
        assert "diversity_a" in report
        assert "frequency_table" in report

    def test_with_comparison(self, sample_smiles):
        report = scaffold_analysis(
            sample_smiles[:5],
            smiles_b=sample_smiles[5:],
        )
        assert "overlap" in report
        assert "jaccard_overlap" in report

    def test_with_target(self, sample_smiles, sample_targets):
        report = scaffold_analysis(
            sample_smiles,
            target_a=sample_targets,
        )
        assert "per_scaffold_targets" in report
