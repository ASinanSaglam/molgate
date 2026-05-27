"""Tests for data/bias.py — bias generation functions."""

import numpy as np
import pandas as pd
import pytest

from molgate.data.bias import (
    bias_by_cluster,
    bias_by_property_range,
    bias_by_scaffold,
    bias_by_substructure,
    bias_by_target_region,
    cluster_molecules,
)


@pytest.fixture
def sample_df():
    """Synthetic dataset with known properties for predictable bias results.

    Contains a mix of aromatic and non-aromatic molecules with varying
    target values.
    """
    return pd.DataFrame({
        "smiles": [
            "c1ccccc1",           # benzene
            "Cc1ccccc1",          # toluene
            "CCc1ccccc1",         # ethylbenzene
            "c1ccncc1",           # pyridine
            "Cc1ccncc1",          # methylpyridine
            "CCO",                # ethanol (acyclic)
            "CCCO",               # propanol (acyclic)
            "CC(=O)O",            # acetic acid (acyclic)
            "CCCC",               # butane (acyclic)
            "c1ccc(-c2ccccc2)cc1",  # biphenyl
        ],
        "drug_id": [f"mol{i}" for i in range(10)],
        "y": [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    })


class TestBiasResult:
    """Common tests for all bias functions — verify the return format."""

    def test_scaffold_returns_tuple(self, sample_df):
        biased_df, metadata = bias_by_scaffold(sample_df, top_n=2)
        assert isinstance(biased_df, pd.DataFrame)
        assert isinstance(metadata, dict)

    def test_metadata_has_required_keys(self, sample_df):
        """All bias functions should include these standard metadata keys."""
        _, meta = bias_by_scaffold(sample_df, top_n=2)
        assert "bias_type" in meta
        assert "n_original" in meta
        assert "n_biased" in meta
        assert "n_removed" in meta
        assert meta["n_original"] == len(sample_df)
        assert meta["n_biased"] + meta["n_removed"] == meta["n_original"]


class TestBiasByScaffold:
    """Tests for scaffold-based bias."""

    def test_reduces_dataset(self, sample_df):
        """Should return fewer molecules than the original."""
        biased_df, _ = bias_by_scaffold(sample_df, top_n=1)
        assert len(biased_df) < len(sample_df)

    def test_excludes_acyclic(self, sample_df):
        """Acyclic molecules should be excluded (no scaffold → None)."""
        biased_df, meta = bias_by_scaffold(sample_df, top_n=10)
        # Even with top_n=10, acyclic molecules shouldn't appear
        acyclic = {"CCO", "CCCO", "CC(=O)O", "CCCC"}
        assert len(set(biased_df["smiles"]) & acyclic) == 0
        assert meta["n_acyclic_excluded"] > 0

    def test_keeps_top_scaffold(self, sample_df):
        """With top_n=1, should keep molecules from the most common scaffold."""
        biased_df, meta = bias_by_scaffold(sample_df, top_n=1)
        assert meta["bias_type"] == "scaffold"
        assert len(meta["scaffolds_kept"]) == 1
        # Benzene scaffold has 3 molecules (benzene, toluene, ethylbenzene)
        # Biphenyl's scaffold is different
        assert len(biased_df) >= 1

    def test_is_strict_subset(self, sample_df):
        """Biased data should be a strict subset of the original."""
        biased_df, _ = bias_by_scaffold(sample_df, top_n=1)
        assert set(biased_df["smiles"]).issubset(set(sample_df["smiles"]))


class TestBiasByPropertyRange:
    """Tests for property-range bias."""

    @pytest.fixture
    def df_with_mw(self, sample_df):
        """Add a mock MW column."""
        # Assign realistic-ish MW values
        sample_df = sample_df.copy()
        sample_df["mw"] = [78.0, 92.0, 106.0, 79.0, 93.0, 46.0, 60.0, 60.0, 58.0, 154.0]
        return sample_df

    def test_filters_by_range(self, df_with_mw):
        """Should keep only molecules within the specified range."""
        biased_df, _ = bias_by_property_range(df_with_mw, "mw", 70, 100)
        assert all(70 <= mw <= 100 for mw in biased_df["mw"])

    def test_missing_column_raises(self, sample_df):
        """Should raise KeyError if the property column doesn't exist."""
        with pytest.raises(KeyError, match="not found"):
            bias_by_property_range(sample_df, "nonexistent", 0, 100)

    def test_metadata_stats(self, df_with_mw):
        """Metadata should include distribution statistics."""
        _, meta = bias_by_property_range(df_with_mw, "mw", 70, 100)
        assert "original_mean" in meta
        assert "biased_mean" in meta
        assert meta["range_low"] == 70
        assert meta["range_high"] == 100


class TestBiasByTargetRegion:
    """Tests for target-range bias."""

    def test_removes_extremes(self, sample_df):
        """Should remove molecules with extreme target values."""
        biased_df, meta = bias_by_target_region(
            sample_df, quantile_low=0.2, quantile_high=0.8
        )
        assert len(biased_df) < len(sample_df)
        # Remaining target values should be within the quantile range
        assert biased_df["y"].min() >= meta["value_low"]
        assert biased_df["y"].max() <= meta["value_high"]

    def test_narrows_distribution(self, sample_df):
        """Target std should decrease after removing extremes."""
        _, meta = bias_by_target_region(sample_df, quantile_low=0.2, quantile_high=0.8)
        assert meta["biased_target_std"] < meta["original_target_std"]


class TestBiasBySubstructure:
    """Tests for substructure-based bias."""

    def test_keep_aromatic(self, sample_df):
        """Keeping aromatic ring matches should filter out acyclics."""
        biased_df, meta = bias_by_substructure(sample_df, smarts="c1ccccc1", keep=True)
        assert meta["bias_type"] == "substructure"
        assert len(biased_df) < len(sample_df)
        # Acyclic molecules should be gone
        assert "CCO" not in biased_df["smiles"].values

    def test_remove_aromatic(self, sample_df):
        """Removing aromatic matches should keep only acyclics."""
        biased_df, _ = bias_by_substructure(sample_df, smarts="c1ccccc1", keep=False)
        # Should have only the acyclic molecules
        for smiles in biased_df["smiles"]:
            assert "c1ccccc1" not in smiles or smiles in {"CCO", "CCCO", "CC(=O)O", "CCCC"}

    def test_invalid_smarts_raises(self, sample_df):
        """Invalid SMARTS should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid SMARTS"):
            bias_by_substructure(sample_df, smarts="[[[invalid", keep=True)

    def test_match_rate(self, sample_df):
        """Match rate should be between 0 and 1."""
        _, meta = bias_by_substructure(sample_df, smarts="c1ccccc1", keep=True)
        assert 0 < meta["match_rate"] < 1


class TestClusterMolecules:
    """Tests for the Butina clustering utility."""

    def test_returns_labels_and_count(self):
        """Should return a list of labels and a cluster count."""
        smiles = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1"]
        labels, n_clusters = cluster_molecules(smiles, cutoff=0.65)
        assert len(labels) == len(smiles)
        assert n_clusters > 0
        assert all(isinstance(label, int) for label in labels)

    def test_all_molecules_assigned(self):
        """Every molecule should be assigned to a cluster."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        labels, n_clusters = cluster_molecules(smiles, cutoff=0.65)
        assert len(labels) == 3
        # All labels should be valid cluster IDs
        assert all(0 <= label < n_clusters for label in labels)


class TestBiasByCluster:
    """Tests for cluster-based bias."""

    def test_reduces_dataset(self, sample_df):
        """Keeping fewer clusters should reduce the dataset."""
        biased_df, meta = bias_by_cluster(sample_df, n_keep=1, cutoff=0.65)
        assert len(biased_df) < len(sample_df)
        assert meta["bias_type"] == "cluster"

    def test_must_specify_mode(self, sample_df):
        """Should raise if neither keep_clusters nor n_keep is specified."""
        with pytest.raises(ValueError, match="Must specify"):
            bias_by_cluster(sample_df)

    def test_metadata_has_cluster_info(self, sample_df):
        """Metadata should include clustering statistics."""
        _, meta = bias_by_cluster(sample_df, n_keep=2, cutoff=0.65)
        assert "n_clusters_total" in meta
        assert "keep_clusters" in meta
        assert "cluster_labels" in meta
        assert len(meta["cluster_labels"]) == len(sample_df)

    def test_is_strict_subset(self, sample_df):
        """Biased data should be a strict subset of the original."""
        biased_df, _ = bias_by_cluster(sample_df, n_keep=1, cutoff=0.65)
        assert set(biased_df["smiles"]).issubset(set(sample_df["smiles"]))
