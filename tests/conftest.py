# conftest.py — shared pytest fixtures for molgate tests.

import numpy as np
import pandas as pd
import pytest


# A small set of real, parseable SMILES covering diverse structures.
# Used across analysis tests to avoid repeating molecule lists.
SAMPLE_SMILES = [
    "CCO",                          # ethanol (acyclic)
    "c1ccccc1",                     # benzene
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen
    "c1ccc(O)cc1",                  # phenol
    "CC(=O)O",                      # acetic acid (acyclic)
    "c1ccncc1",                     # pyridine
    "C1CCCCC1",                     # cyclohexane
    "c1ccc(-c2ccccc2)cc1",          # biphenyl
    "CC(C)(C)c1ccc(O)cc1",          # 4-tert-butylphenol
]

# Fake target values (one per SMILES above)
SAMPLE_TARGETS = np.array([-1.0, -2.5, -1.5, -3.0, -0.5, 0.5, -2.0, -4.0, -3.5, -1.8])


@pytest.fixture
def sample_smiles():
    """10 diverse SMILES strings."""
    return SAMPLE_SMILES.copy()


@pytest.fixture
def sample_targets():
    """Target values matching sample_smiles."""
    return SAMPLE_TARGETS.copy()


@pytest.fixture
def sample_df():
    """DataFrame matching the output format of load_dataset."""
    return pd.DataFrame({
        "smiles": SAMPLE_SMILES,
        "drug_id": [f"mol_{i}" for i in range(len(SAMPLE_SMILES))],
        "y": SAMPLE_TARGETS,
    })
