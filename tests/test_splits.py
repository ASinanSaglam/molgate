"""Tests for data/splits.py — random, scaffold, and stratified splitting."""

import numpy as np
import pandas as pd
import pytest

from molgate.data.splits import random_split, scaffold_split, stratified_split


@pytest.fixture
def sample_df():
    """Create a synthetic dataset with a mix of scaffolded and acyclic molecules.

    Includes molecules with known scaffolds so scaffold_split behavior
    is predictable.
    """
    smiles = [
        # Benzene scaffold group (3 molecules)
        "c1ccccc1",           # benzene
        "Cc1ccccc1",          # toluene
        "CCc1ccccc1",         # ethylbenzene
        # Pyridine scaffold group (2 molecules)
        "c1ccncc1",           # pyridine
        "Cc1ccncc1",          # 2-methylpyridine
        # Acyclic molecules (no scaffold)
        "CCO",                # ethanol
        "CCCO",               # propanol
        "CC(=O)O",            # acetic acid
        "CCCC",               # butane
        "CC(C)C",             # isobutane
    ]
    return pd.DataFrame({
        "smiles": smiles,
        "drug_id": [f"mol{i}" for i in range(len(smiles))],
        "y": np.linspace(-5.0, 5.0, len(smiles)),
    })


class TestRandomSplit:
    """Tests for random_split."""

    def test_returns_three_splits(self, sample_df):
        """Should return dict with train, val, test keys."""
        splits = random_split(sample_df)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_no_data_loss(self, sample_df):
        """Total molecules across splits should equal original."""
        splits = random_split(sample_df)
        total = sum(len(splits[k]) for k in ["train", "val", "test"])
        assert total == len(sample_df)

    def test_no_overlap(self, sample_df):
        """No molecule should appear in more than one split."""
        splits = random_split(sample_df)
        train_smiles = set(splits["train"]["smiles"])
        val_smiles = set(splits["val"]["smiles"])
        test_smiles = set(splits["test"]["smiles"])
        assert len(train_smiles & val_smiles) == 0
        assert len(train_smiles & test_smiles) == 0
        assert len(val_smiles & test_smiles) == 0

    def test_reproducible(self, sample_df):
        """Same seed should produce the same split."""
        s1 = random_split(sample_df, seed=42)
        s2 = random_split(sample_df, seed=42)
        pd.testing.assert_frame_equal(s1["train"], s2["train"])
        pd.testing.assert_frame_equal(s1["test"], s2["test"])

    def test_different_seeds(self, sample_df):
        """Different seeds should produce different splits."""
        s1 = random_split(sample_df, seed=42)
        s2 = random_split(sample_df, seed=123)
        # Very unlikely to be identical with different seeds
        assert not s1["train"]["smiles"].equals(s2["train"]["smiles"])

    def test_index_reset(self, sample_df):
        """Each split DataFrame should have a clean 0..N-1 index."""
        splits = random_split(sample_df)
        for key in ["train", "val", "test"]:
            assert list(splits[key].index) == list(range(len(splits[key])))


class TestScaffoldSplit:
    """Tests for scaffold_split."""

    def test_returns_three_splits(self, sample_df):
        """Should return dict with train, val, test keys."""
        splits = scaffold_split(sample_df)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_no_data_loss(self, sample_df):
        """Total molecules across splits should equal original."""
        splits = scaffold_split(sample_df)
        total = sum(len(splits[k]) for k in ["train", "val", "test"])
        assert total == len(sample_df)

    def test_no_overlap(self, sample_df):
        """No molecule should appear in more than one split."""
        splits = scaffold_split(sample_df)
        all_smiles = []
        for key in ["train", "val", "test"]:
            all_smiles.extend(splits[key]["smiles"].tolist())
        assert len(all_smiles) == len(set(all_smiles))

    def test_reproducible(self, sample_df):
        """Same seed should produce the same split."""
        s1 = scaffold_split(sample_df, seed=42)
        s2 = scaffold_split(sample_df, seed=42)
        pd.testing.assert_frame_equal(s1["train"], s2["train"])

    def test_train_is_largest(self, sample_df):
        """Training set should be the largest split."""
        splits = scaffold_split(sample_df, val_frac=0.1, test_frac=0.1)
        assert len(splits["train"]) >= len(splits["val"])
        assert len(splits["train"]) >= len(splits["test"])


class TestStratifiedSplit:
    """Tests for stratified_split."""

    def test_returns_three_splits(self, sample_df):
        """Should return dict with train, val, test keys."""
        splits = stratified_split(sample_df)
        assert set(splits.keys()) == {"train", "val", "test"}

    def test_no_data_loss(self, sample_df):
        """Total molecules across splits should equal original."""
        splits = stratified_split(sample_df)
        total = sum(len(splits[k]) for k in ["train", "val", "test"])
        assert total == len(sample_df)

    def test_no_overlap(self, sample_df):
        """No molecule should appear in more than one split."""
        splits = stratified_split(sample_df)
        all_smiles = []
        for key in ["train", "val", "test"]:
            all_smiles.extend(splits[key]["smiles"].tolist())
        assert len(all_smiles) == len(set(all_smiles))

    def test_target_distribution_preserved(self):
        """Train and test should have similar target distribution means."""
        # Larger dataset for meaningful stratification
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "smiles": [f"C{'C' * i}" for i in range(n)],
            "drug_id": [f"mol{i}" for i in range(n)],
            "y": rng.normal(0, 3, n),
        })
        splits = stratified_split(df, n_bins=10, seed=42)
        train_mean = splits["train"]["y"].mean()
        test_mean = splits["test"]["y"].mean()
        # Means should be within 0.5 of each other (loose check)
        assert abs(train_mean - test_mean) < 0.5

    def test_handles_classification(self):
        """Should work with binary target values (classification)."""
        df = pd.DataFrame({
            "smiles": [f"C{'C' * i}" for i in range(50)],
            "drug_id": [f"mol{i}" for i in range(50)],
            "y": [0] * 30 + [1] * 20,
        })
        # duplicates="drop" in pd.qcut should handle this
        splits = stratified_split(df, n_bins=10, seed=42)
        total = sum(len(splits[k]) for k in ["train", "val", "test"])
        assert total == 50
