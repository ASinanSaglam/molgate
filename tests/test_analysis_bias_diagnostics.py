"""Tests for analysis/bias_diagnostics.py."""

import numpy as np
import pytest

from molgate.analysis.bias_diagnostics import (
    adversarial_validation,
    bias_report,
    descriptor_shift,
    diversity_change,
    target_shift,
)


class TestDescriptorShift:
    """Tests for descriptor distribution shift."""

    def test_returns_dataframe(self, sample_smiles):
        result = descriptor_shift(sample_smiles, sample_smiles[:5])
        assert "ks_statistic" in result.columns
        assert "significant" in result.columns

    def test_identical_no_shift(self, sample_smiles):
        """Comparing a dataset to itself should show zero shift."""
        result = descriptor_shift(sample_smiles, sample_smiles)
        assert (result["ks_statistic"] == 0.0).all()


class TestDiversityChange:
    """Tests for scaffold diversity comparison."""

    def test_returns_expected_keys(self, sample_smiles):
        result = diversity_change(sample_smiles, sample_smiles[:5])
        assert "scaffold_ratio_original" in result
        assert "scaffold_ratio_biased" in result
        assert "scaffold_ratio_change" in result
        assert "n_molecules_original" in result

    def test_same_dataset_zero_change(self, sample_smiles):
        result = diversity_change(sample_smiles, sample_smiles)
        assert result["scaffold_ratio_change"] == pytest.approx(0.0)
        assert result["singleton_fraction_change"] == pytest.approx(0.0)


class TestTargetShift:
    """Tests for target distribution comparison."""

    def test_returns_expected_keys(self, sample_targets):
        result = target_shift(sample_targets, sample_targets[:5])
        expected = {
            "mean_original", "mean_biased", "mean_change",
            "std_original", "std_biased", "std_change",
            "ks_statistic", "ks_p_value", "target_shift_significant",
        }
        assert expected.issubset(set(result.keys()))

    def test_same_target_no_shift(self, sample_targets):
        result = target_shift(sample_targets, sample_targets)
        assert result["ks_statistic"] == pytest.approx(0.0)
        assert result["mean_change"] == pytest.approx(0.0)

    def test_narrower_range(self, sample_targets):
        """Removing extremes should reduce range."""
        sorted_t = np.sort(sample_targets)
        # Drop lowest and highest
        trimmed = sorted_t[1:-1]
        result = target_shift(sample_targets, trimmed)
        assert result["range_biased"] <= result["range_original"]


class TestAdversarialValidation:
    """Tests for adversarial validation."""

    def test_returns_expected_keys(self, sample_smiles):
        result = adversarial_validation(
            sample_smiles[:5], sample_smiles[5:],
            n_folds=2,  # fewer folds for speed with small data
        )
        assert "auroc_mean" in result
        assert "auroc_std" in result
        assert "distinguishable" in result

    def test_auroc_bounded(self, sample_smiles):
        result = adversarial_validation(
            sample_smiles[:5], sample_smiles[5:],
            n_folds=2,
        )
        assert 0.0 <= result["auroc_mean"] <= 1.0

    def test_identical_sets_low_auroc(self):
        """Same molecules in both sets → classifier should struggle."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1", "C1CCCCC1"] * 4
        result = adversarial_validation(smiles, smiles, n_folds=2)
        # AUROC should be near 0.5 (can't distinguish identical sets)
        assert result["auroc_mean"] < 0.8


class TestBiasReport:
    """Tests for the full bias_report function."""

    def test_returns_expected_keys(self, sample_smiles, sample_targets):
        report = bias_report(
            sample_smiles, sample_smiles[:5],
            sample_targets, sample_targets[:5],
            bias_name="test_bias",
            run_adversarial=False,  # skip for speed
        )
        assert report["bias_name"] == "test_bias"
        assert "retention_rate" in report
        assert "descriptor_shift" in report
        assert "diversity_change" in report
        assert "target_shift" in report
        assert "n_significant_descriptors" in report

    def test_retention_rate(self, sample_smiles, sample_targets):
        report = bias_report(
            sample_smiles, sample_smiles[:5],
            sample_targets, sample_targets[:5],
            run_adversarial=False,
        )
        assert report["retention_rate"] == pytest.approx(0.5)

    def test_with_adversarial(self, sample_smiles, sample_targets):
        report = bias_report(
            sample_smiles, sample_smiles[:5],
            sample_targets, sample_targets[:5],
            run_adversarial=True,
        )
        assert "adversarial_auroc" in report
        assert "adversarial_distinguishable" in report
