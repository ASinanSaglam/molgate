"""Tests for data/loaders.py — SMILES canonicalization and cleaning logic.

We test the internal functions with synthetic data to avoid TDC network calls.
The full load_dataset integration is tested with a small real dataset
(marked slow so it can be skipped in fast CI runs).
"""

import pandas as pd
import pytest

from molgate.data.loaders import _canonicalize_smiles, _clean_dataframe
from molgate.data.registry import get_dataset_info


class TestCanonicalizeSmiles:
    """Tests for the _canonicalize_smiles helper."""

    def test_canonical_form(self):
        """Different SMILES for the same molecule should produce the same output."""
        # Ethanol written three ways
        assert _canonicalize_smiles("CCO") == _canonicalize_smiles("OCC")
        assert _canonicalize_smiles("CCO") == _canonicalize_smiles("C(O)C")

    def test_already_canonical(self):
        """Canonical SMILES should be returned unchanged."""
        canonical = "c1ccccc1"  # benzene
        assert _canonicalize_smiles(canonical) == canonical

    def test_invalid_smiles(self):
        """Invalid SMILES should return None."""
        assert _canonicalize_smiles("not_a_molecule") is None

    def test_complex_molecule(self):
        """Aspirin should canonicalize consistently."""
        result = _canonicalize_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


class TestCleanDataframe:
    """Tests for the _clean_dataframe function using synthetic data."""

    @pytest.fixture
    def raw_df(self):
        """Create a synthetic DataFrame mimicking TDC's output format."""
        return pd.DataFrame({
            "Drug": ["CCO", "OCC", "c1ccccc1", "invalid_smiles", "CC(=O)O"],
            "Drug_ID": ["mol1", "mol2", "mol3", "mol4", "mol5"],
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

    @pytest.fixture
    def sol_info(self):
        """Get solubility DatasetInfo for column name reference."""
        return get_dataset_info("solubility")

    def test_column_renaming(self, raw_df, sol_info):
        """Columns should be renamed from TDC convention to ours."""
        result = _clean_dataframe(raw_df, sol_info)
        assert list(result.columns) == ["smiles", "drug_id", "y"]

    def test_invalid_smiles_removed(self, raw_df, sol_info):
        """Rows with unparseable SMILES should be dropped."""
        result = _clean_dataframe(raw_df, sol_info)
        # "invalid_smiles" should be gone
        assert len(result) < len(raw_df)
        assert "invalid_smiles" not in result["smiles"].values

    def test_duplicates_removed(self, raw_df, sol_info):
        """Duplicate SMILES (after canonicalization) should be deduplicated."""
        result = _clean_dataframe(raw_df, sol_info)
        # "CCO" and "OCC" are the same molecule — only one should remain
        ethanol_canonical = _canonicalize_smiles("CCO")
        ethanol_rows = result[result["smiles"] == ethanol_canonical]
        assert len(ethanol_rows) == 1

    def test_keeps_first_duplicate(self, raw_df, sol_info):
        """When deduplicating, the first occurrence should be kept."""
        result = _clean_dataframe(raw_df, sol_info)
        ethanol_canonical = _canonicalize_smiles("CCO")
        ethanol_row = result[result["smiles"] == ethanol_canonical].iloc[0]
        # mol1 was first
        assert ethanol_row["drug_id"] == "mol1"
        assert ethanol_row["y"] == 1.0

    def test_index_reset(self, raw_df, sol_info):
        """Output index should be 0..N-1."""
        result = _clean_dataframe(raw_df, sol_info)
        assert list(result.index) == list(range(len(result)))

    def test_all_valid_no_loss(self, sol_info):
        """A DataFrame with all valid, unique SMILES should lose nothing."""
        df = pd.DataFrame({
            "Drug": ["CCO", "c1ccccc1", "CC(=O)O"],
            "Drug_ID": ["a", "b", "c"],
            "Y": [1.0, 2.0, 3.0],
        })
        result = _clean_dataframe(df, sol_info)
        assert len(result) == 3


@pytest.mark.slow
class TestLoadDataset:
    """Integration tests that hit the TDC network. Marked slow."""

    def test_load_solubility(self):
        """load_dataset should return a clean DataFrame for solubility."""
        from molgate.data.loaders import load_dataset

        df = load_dataset("solubility")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["smiles", "drug_id", "y"]
        assert len(df) > 5000  # solubility has ~9982 compounds
        # No duplicates in canonical SMILES
        assert df["smiles"].is_unique
        # No NaN values
        assert df.notna().all().all()
