"""Tests for data/featurizer.py — descriptor, fingerprint, and graph featurization."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data as PyGData

from molgate.data.featurizer import (
    DESCRIPTOR_LIST,
    compute_descriptors,
    compute_fingerprints,
    smiles_to_graph,
    smiles_list_to_graphs,
)

# Small set of valid SMILES for testing
SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]


class TestComputeDescriptors:
    """Tests for RDKit 2D descriptor computation."""

    def test_output_shape(self):
        """Should return DataFrame with n_molecules rows and n_descriptors columns."""
        df = compute_descriptors(SAMPLE_SMILES)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(SAMPLE_SMILES)
        assert len(df.columns) == len(DESCRIPTOR_LIST)

    def test_column_names(self):
        """Column names should match DESCRIPTOR_LIST names."""
        df = compute_descriptors(SAMPLE_SMILES)
        expected_names = [name for name, _ in DESCRIPTOR_LIST]
        assert list(df.columns) == expected_names

    def test_no_nans_for_valid_smiles(self):
        """Valid SMILES should produce no NaN values."""
        df = compute_descriptors(SAMPLE_SMILES)
        assert df.notna().all().all()

    def test_molecular_weight_reasonable(self):
        """MW for ethanol (46.07) should be in a reasonable range."""
        df = compute_descriptors(["CCO"])
        assert 40 < df["mw"].iloc[0] < 50

    def test_benzene_is_aromatic(self):
        """Benzene should have 1 aromatic ring."""
        df = compute_descriptors(["c1ccccc1"])
        assert df["aromatic_rings"].iloc[0] == 1

    def test_single_molecule(self):
        """Should work for a single molecule."""
        df = compute_descriptors(["CCO"])
        assert len(df) == 1


class TestComputeFingerprints:
    """Tests for Morgan fingerprint computation."""

    def test_output_shape(self):
        """Should return array of shape (n_molecules, n_bits)."""
        fps = compute_fingerprints(SAMPLE_SMILES, radius=2, n_bits=2048)
        assert isinstance(fps, np.ndarray)
        assert fps.shape == (len(SAMPLE_SMILES), 2048)

    def test_dtype(self):
        """Output should be int8 (binary values)."""
        fps = compute_fingerprints(SAMPLE_SMILES)
        assert fps.dtype == np.int8

    def test_binary_values(self):
        """All values should be 0 or 1."""
        fps = compute_fingerprints(SAMPLE_SMILES)
        assert set(np.unique(fps)).issubset({0, 1})

    def test_nonzero_bits(self):
        """Each molecule should have some bits set."""
        fps = compute_fingerprints(SAMPLE_SMILES)
        for i in range(len(SAMPLE_SMILES)):
            assert fps[i].sum() > 0

    def test_custom_nbits(self):
        """n_bits parameter should control output width."""
        fps = compute_fingerprints(SAMPLE_SMILES, n_bits=512)
        assert fps.shape[1] == 512

    def test_identical_molecules_same_fingerprint(self):
        """Same molecule (different SMILES) should produce identical fingerprints."""
        fps = compute_fingerprints(["CCO", "OCC"])  # both are ethanol
        # After canonicalization they should be the same
        # (fingerprints are computed from the mol object, not the SMILES string)
        np.testing.assert_array_equal(fps[0], fps[1])


class TestSmilesToGraph:
    """Tests for PyG graph conversion."""

    def test_returns_data_object(self):
        """Should return a PyG Data object."""
        graph = smiles_to_graph("CCO")
        assert isinstance(graph, PyGData)

    def test_node_features(self):
        """Ethanol (CCO) has 3 heavy atoms → 3 nodes, each with 14 features."""
        graph = smiles_to_graph("CCO")
        assert graph.x.shape == (3, 14)
        assert graph.x.dtype == torch.float

    def test_edge_features(self):
        """Ethanol has 2 bonds → 4 directed edges, each with 6 features."""
        graph = smiles_to_graph("CCO")
        assert graph.edge_index.shape[0] == 2  # COO format: 2 rows
        assert graph.edge_index.shape[1] == 4  # 2 bonds × 2 directions
        assert graph.edge_attr.shape == (4, 6)

    def test_bidirectional_edges(self):
        """Every edge should appear in both directions."""
        graph = smiles_to_graph("CCO")
        edges = set()
        for col in range(graph.edge_index.shape[1]):
            src, dst = graph.edge_index[0, col].item(), graph.edge_index[1, col].item()
            edges.add((src, dst))
        # For each (i,j), (j,i) should also exist
        for src, dst in list(edges):
            assert (dst, src) in edges

    def test_y_attached(self):
        """Target value should be attached when provided."""
        graph = smiles_to_graph("CCO", y=1.5)
        assert graph.y is not None
        assert graph.y.item() == pytest.approx(1.5)

    def test_y_not_attached(self):
        """No target when y is None."""
        graph = smiles_to_graph("CCO")
        assert not hasattr(graph, "y") or graph.y is None

    def test_smiles_stored(self):
        """Original SMILES should be stored on the graph."""
        graph = smiles_to_graph("CCO")
        assert graph.smiles == "CCO"

    def test_invalid_smiles_returns_none(self):
        """Invalid SMILES should return None."""
        assert smiles_to_graph("not_a_molecule") is None

    def test_benzene_ring(self):
        """Benzene (c1ccccc1) should have 6 atoms and 6 bonds (12 edges)."""
        graph = smiles_to_graph("c1ccccc1")
        assert graph.num_nodes == 6
        assert graph.num_edges == 12  # 6 bonds × 2 directions


class TestSmilesListToGraphs:
    """Tests for batch graph conversion."""

    def test_batch_conversion(self):
        """Should convert a list of SMILES to a list of graphs."""
        graphs = smiles_list_to_graphs(SAMPLE_SMILES)
        assert len(graphs) == len(SAMPLE_SMILES)
        assert all(isinstance(g, PyGData) for g in graphs)

    def test_with_targets(self):
        """Should attach target values when provided."""
        y_vals = [1.0, 2.0, 3.0, 4.0]
        graphs = smiles_list_to_graphs(SAMPLE_SMILES, y_list=y_vals)
        for g, y in zip(graphs, y_vals):
            assert g.y.item() == pytest.approx(y)

    def test_skips_invalid(self):
        """Invalid SMILES should be skipped with a warning."""
        mixed = ["CCO", "not_valid", "c1ccccc1"]
        graphs = smiles_list_to_graphs(mixed)
        assert len(graphs) == 2  # only the 2 valid ones
