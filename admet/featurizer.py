"""Molecular featurizer wrapping DeepChem's MolGraphConvFeaturizer.

Produces PyG Data objects with 30-dim atom features and 11-dim bond features,
matching MolGraphConvFeaturizer's fixed output dimensions. An RDKit-only fallback
is provided for environments where DeepChem is not installed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data


# MolGraphConvFeaturizer fixed output dimensions
ATOM_FEATURE_DIM = 30
BOND_FEATURE_DIM = 11


class MolFeaturizer(Protocol):
    """Protocol for molecular featurizers."""

    def featurize(self, smiles: str) -> Data | None:
        """Featurize a SMILES string to a PyG Data object.

        Returns None if the SMILES cannot be parsed or featurized.
        """
        ...


class DeepChemFeaturizer:
    """Wraps DeepChem's MolGraphConvFeaturizer to produce PyG Data objects.

    Atom features (30-dim) and bond features (11-dim) are the fixed output of
    MolGraphConvFeaturizer. See DeepChem docs for the full feature list.
    """

    def __init__(self) -> None:
        try:
            from deepchem.feat import MolGraphConvFeaturizer
        except ImportError as e:
            raise ImportError(
                "DeepChem is required for DeepChemFeaturizer. "
                "Install it with: pip install deepchem\n"
                "Or use RDKitFeaturizer for an RDKit-only alternative."
            ) from e
        self._featurizer = MolGraphConvFeaturizer(use_edges=True)

    def featurize(self, smiles: str) -> Data | None:
        """Featurize a SMILES string to a PyG Data object.

        Returns None if the molecule cannot be parsed or featurized.
        DeepChem returns an empty numpy array (not None) for failed molecules,
        so we check for the GraphData type explicitly.
        """
        from deepchem.feat.graph_data import GraphData

        result = self._featurizer.featurize([smiles])
        if result is None or len(result) == 0:
            return None
        graph = result[0]
        if not isinstance(graph, GraphData):
            return None

        x = torch.tensor(graph.node_features, dtype=torch.float)
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(graph.edge_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# RDKit-only fallback — produces tensors with the same shape as DeepChem's
# MolGraphConvFeaturizer so the GNN backbone works without DeepChem installed.
# ---------------------------------------------------------------------------

_ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
                 "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
                 "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H"]
_ATOM_SYMBOL_IDX: dict[str, int] = {s: i for i, s in enumerate(_ATOM_SYMBOLS)}

_HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _atom_features(atom: Chem.Atom) -> list[float]:
    """Compute 30-dim atom feature vector matching MolGraphConvFeaturizer layout.

    Layout (matches DeepChem):
    - one-hot atom type (30 elements, last = other): 30 dims
    Total: 30 dims

    Note: This is a simplified approximation. The full MolGraphConvFeaturizer
    includes chirality, formal charge, partial charges, etc. The RDKit fallback
    is for smoke-testing the pipeline; use DeepChemFeaturizer for real training.
    """
    symbol = atom.GetSymbol()
    idx = _ATOM_SYMBOL_IDX.get(symbol, len(_ATOM_SYMBOLS) - 1)
    one_hot = [0.0] * ATOM_FEATURE_DIM
    one_hot[idx] = 1.0
    return one_hot


def _bond_features(bond: Chem.Bond) -> list[float]:
    """Compute 11-dim bond feature vector.

    Layout (approximates DeepChem):
    - bond type one-hot (4 dims)
    - is conjugated (1)
    - is in ring (1)
    - stereo one-hot (6 dims, simplified to 5 + padding)
    Total: 11 dims
    """
    bt = bond.GetBondType()
    bond_type_oh = [float(bt == t) for t in _BOND_TYPES]  # 4 dims
    conj = [float(bond.GetIsConjugated())]                 # 1 dim
    in_ring = [float(bond.IsInRing())]                     # 1 dim
    # Stereo: 6 types in DeepChem, simplified here to 5 + pad
    stereo = int(bond.GetStereo())
    stereo_oh = [float(stereo == i) for i in range(5)]     # 5 dims
    return bond_type_oh + conj + in_ring + stereo_oh        # 4+1+1+5 = 11


class RDKitFeaturizer:
    """RDKit-only fallback featurizer producing (30, 11)-dim features.

    Use this when DeepChem is not installed. Feature dimensions match
    DeepChemFeaturizer so the GNN backbone is compatible with both.
    For final training and benchmarking, prefer DeepChemFeaturizer.
    """

    def featurize(self, smiles: str) -> Data | None:
        """Featurize a SMILES string to a PyG Data object.

        Returns None if the SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Node features
        atom_feats = [_atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)  # (N, 30)

        # Edge features — undirected: add both directions
        edge_indices: list[list[int]] = [[], []]
        edge_feats: list[list[float]] = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feats = _bond_features(bond)
            edge_indices[0] += [i, j]
            edge_indices[1] += [j, i]
            edge_feats += [feats, feats]

        if edge_feats:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        else:
            # Isolated atoms (single-atom molecules)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def get_featurizer(backend: str = "auto") -> MolFeaturizer:
    """Return a featurizer instance.

    Args:
        backend: "deepchem" forces DeepChemFeaturizer (raises if not installed),
                 "rdkit" forces RDKitFeaturizer,
                 "auto" tries DeepChem first, falls back to RDKit.
    """
    if backend == "deepchem":
        return DeepChemFeaturizer()
    if backend == "rdkit":
        return RDKitFeaturizer()
    # auto
    try:
        return DeepChemFeaturizer()
    except ImportError:
        return RDKitFeaturizer()
