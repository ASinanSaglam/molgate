"""Shared fixtures for the test suite."""

from __future__ import annotations

import pandas as pd
import pytest
from rdkit import Chem


SAMPLE_SMILES = [
    "CCO",                      # ethanol
    "c1ccccc1",                 # benzene
    "CC(=O)O",                  # acetic acid
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc(cc1)C(=O)O",        # benzoic acid
    "CCCCC",                    # pentane
]


@pytest.fixture
def sample_smiles() -> list[str]:
    return SAMPLE_SMILES


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Small regression DataFrame mimicking TDC format (Drug, Y columns)."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Drug": SAMPLE_SMILES,
        "Y": rng.uniform(-4.0, 2.0, len(SAMPLE_SMILES)),
    })


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """Small binary classification DataFrame mimicking TDC format."""
    return pd.DataFrame({
        "Drug": SAMPLE_SMILES,
        "Y": [1, 0, 1, 0, 1, 0, 1, 0],
    })


@pytest.fixture
def rdkit_featurizer():
    from admet.featurizer import RDKitFeaturizer
    return RDKitFeaturizer()
