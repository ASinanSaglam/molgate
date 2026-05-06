"""Tests for admet/featurizer.py."""

from __future__ import annotations

import pytest
import torch


def test_rdkit_featurizer_output_shape(rdkit_featurizer):
    data = rdkit_featurizer.featurize("CCO")
    assert data is not None
    assert data.x.shape == (3, 30), f"Expected (3, 30), got {data.x.shape}"
    assert data.edge_attr.shape[1] == 11
    assert data.edge_index.shape[0] == 2


def test_rdkit_featurizer_invalid_smiles(rdkit_featurizer):
    result = rdkit_featurizer.featurize("not_a_smiles_!!!!")
    assert result is None


def test_rdkit_featurizer_single_atom(rdkit_featurizer):
    # Single atom molecule — no bonds
    data = rdkit_featurizer.featurize("[Na+]")
    assert data is not None
    assert data.x.shape[0] == 1
    assert data.edge_index.shape == (2, 0)


def test_rdkit_featurizer_undirected_edges(rdkit_featurizer):
    # Ethanol has 2 bonds → 4 directed edges
    data = rdkit_featurizer.featurize("CCO")
    assert data.edge_index.shape[1] == 4


def test_get_featurizer_rdkit_backend():
    from admet.featurizer import get_featurizer, RDKitFeaturizer
    feat = get_featurizer("rdkit")
    assert isinstance(feat, RDKitFeaturizer)


def test_get_featurizer_auto():
    from admet.featurizer import get_featurizer
    feat = get_featurizer("auto")
    # Should return some featurizer without crashing
    data = feat.featurize("c1ccccc1")
    assert data is not None
    assert data.x.shape[1] == 30
    assert data.edge_attr.shape[1] == 11
