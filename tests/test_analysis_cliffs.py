"""Tests for analysis/cliffs.py."""

import numpy as np
import pytest

from molgate.analysis.cliffs import (
    CliffPair,
    cliff_analysis,
    cliff_summary,
    find_cliffs,
)


class TestFindCliffs:
    """Tests for activity cliff detection."""

    def test_returns_list_of_cliffpairs(self, sample_smiles, sample_targets):
        cliffs = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.3)
        assert isinstance(cliffs, list)
        if cliffs:
            assert isinstance(cliffs[0], CliffPair)

    def test_sorted_by_activity_diff(self, sample_smiles, sample_targets):
        cliffs = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.3)
        if len(cliffs) > 1:
            diffs = [c.activity_diff for c in cliffs]
            assert all(diffs[i] >= diffs[i + 1] for i in range(len(diffs) - 1))

    def test_high_threshold_fewer_cliffs(self, sample_smiles, sample_targets):
        """Higher similarity threshold → fewer similar pairs → fewer cliffs."""
        cliffs_low = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.3)
        cliffs_high = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.9)
        assert len(cliffs_high) <= len(cliffs_low)

    def test_identical_targets_no_cliffs(self, sample_smiles):
        """If all targets are the same, no activity cliffs exist."""
        targets = np.ones(len(sample_smiles))
        cliffs = find_cliffs(sample_smiles, targets, sim_threshold=0.3, act_threshold=0.1)
        assert len(cliffs) == 0

    def test_length_mismatch_raises(self, sample_smiles):
        with pytest.raises(ValueError, match="Length mismatch"):
            find_cliffs(sample_smiles, np.array([1.0, 2.0]))

    def test_cliff_pair_fields(self, sample_smiles, sample_targets):
        """CliffPair should have all expected fields."""
        cliffs = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.3)
        if cliffs:
            c = cliffs[0]
            assert c.idx_a != c.idx_b
            assert c.similarity >= 0.3
            assert c.activity_diff >= 0
            assert c.activity_diff == pytest.approx(abs(c.activity_a - c.activity_b))

    def test_no_similar_pairs_returns_empty(self):
        """Completely different molecules with high threshold → no cliffs."""
        # Very different molecules
        smiles = ["C", "c1ccc2ccccc2c1", "C1CC1"]  # methane, naphthalene, cyclopropane
        targets = np.array([1.0, 5.0, 3.0])
        cliffs = find_cliffs(smiles, targets, sim_threshold=0.99)
        assert len(cliffs) == 0


class TestCliffSummary:
    """Tests for cliff_summary."""

    def test_empty_cliffs(self):
        result = cliff_summary([], n_molecules=100)
        assert result["n_cliff_pairs"] == 0
        assert result["cliff_molecule_fraction"] == 0.0
        assert result["most_dramatic"] is None

    def test_with_cliffs(self, sample_smiles, sample_targets):
        cliffs = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.3)
        if cliffs:
            result = cliff_summary(cliffs, n_molecules=len(sample_smiles))
            assert result["n_cliff_pairs"] == len(cliffs)
            assert result["n_cliff_molecules"] <= len(sample_smiles)
            assert 0 <= result["cliff_molecule_fraction"] <= 1.0
            assert result["most_dramatic"] is cliffs[0]
            assert result["mean_cliff_sim"] >= 0.3

    def test_fraction_bounded(self, sample_smiles, sample_targets):
        cliffs = find_cliffs(sample_smiles, sample_targets, sim_threshold=0.3)
        result = cliff_summary(cliffs, n_molecules=len(sample_smiles))
        assert 0.0 <= result["cliff_molecule_fraction"] <= 1.0


class TestCliffAnalysis:
    """Tests for the convenience cliff_analysis function."""

    def test_returns_expected_keys(self, sample_smiles, sample_targets):
        report = cliff_analysis(sample_smiles, sample_targets, sim_threshold=0.3)
        assert "cliffs" in report
        assert "summary" in report
        assert "n_cliff_pairs" in report
        assert "cliff_molecule_fraction" in report
