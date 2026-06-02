"""Graph Neural Network for molecular property prediction.

This module defines ``MoleculeGNN``, a message-passing GNN that operates
directly on molecular graphs (atoms = nodes, bonds = edges).

Architecture overview::

    SMILES → molecular graph (via featurizer.smiles_to_graph)
                │
        ┌───────▼────────┐
        │  Atom embedding │   Linear: 14 → hidden_dim
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  MP Layer 1     │   GINEConv + BatchNorm + ReLU + Dropout
        │  MP Layer 2     │   (repeated num_layers times)
        │  MP Layer N     │
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  Global pool    │   mean or sum over all atoms → single vector
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │  MLP head       │   hidden_dim → hidden_dim → 1
        └───────┬────────┘
                │
              scalar prediction

Why GINEConv?
    GIN (Graph Isomorphism Network) is provably as powerful as the
    Weisfeiler-Leman graph isomorphism test — it can distinguish any
    pair of non-isomorphic graphs that WL can. The "E" variant (GINE)
    additionally incorporates edge features into the message computation,
    which matters because bond type (single/double/aromatic) carries
    chemical information that a plain GCN/GIN would ignore.

    Compared to GCNConv:
        - GCN averages neighbor features (lossy — can't distinguish
          different multisets with the same mean)
        - GIN sums neighbor features (injective — preserves the full
          multiset structure)

    For molecular property prediction, GIN/GINE consistently matches
    or outperforms GCN in published benchmarks.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Atom embedding: projects raw atom features into the hidden space
# ---------------------------------------------------------------------------

ATOM_FEAT_DIM = 14   # Must match featurizer._atom_features output length
BOND_FEAT_DIM = 6    # Must match featurizer._bond_features output length


class MoleculeGNN(nn.Module):
    """Message-passing GNN for molecular property prediction.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of hidden node representations. All message-passing
        layers, the atom embedding, and the MLP head use this width.
    num_layers : int
        Number of message-passing rounds. Each round aggregates information
        from 1-hop neighbors, so ``num_layers=3`` means each atom "sees"
        atoms up to 3 bonds away. For drug-like molecules (typically
        20-50 atoms, diameter ~10 bonds), 3-4 layers is usually enough.
    task_type : str
        "regression" or "classification". Determines the output activation:
        none for regression (raw scalar), sigmoid for classification
        (probability in [0, 1]).
    dropout : float
        Dropout probability applied after each MP layer. Regularises the
        model by randomly zeroing hidden dimensions during training.
    pool : str
        Global pooling strategy: "mean" or "sum".
        - "mean" averages atom representations → size-invariant
          (large and small molecules produce same-scale vectors).
        - "sum" adds atom representations → size-aware
          (larger molecules produce larger vectors, which can help
          when molecular size correlates with the target property).
        For most ADMET tasks, "mean" works slightly better.
    atom_feat_dim : int
        Input atom feature dimensionality (default: 14, matching our
        featurizer).
    bond_feat_dim : int
        Input bond feature dimensionality (default: 6, matching our
        featurizer).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        task_type: str = "regression",
        dropout: float = 0.1,
        pool: str = "mean",
        atom_feat_dim: int = ATOM_FEAT_DIM,
        bond_feat_dim: int = BOND_FEAT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task_type = task_type
        self.dropout_rate = dropout
        self.pool_name = pool

        # --- Atom embedding ---
        # Projects raw 14-dim atom features into the hidden space.
        # This is a learned linear transformation, not a lookup table —
        # our atom features are continuous (degree, charge) not just IDs.
        self.atom_encoder = nn.Linear(atom_feat_dim, hidden_dim)

        # --- Bond embedding ---
        # Projects 6-dim bond features into the hidden space.
        # GINEConv requires edge features to have the same dimensionality
        # as node features, so we project bonds into hidden_dim too.
        self.bond_encoder = nn.Linear(bond_feat_dim, hidden_dim)

        # --- Message-passing layers ---
        # Each layer is: GINEConv → BatchNorm → ReLU → Dropout
        #
        # We build GINEConv by wrapping a 2-layer MLP as the "update"
        # function. GIN's aggregation is:
        #   h_v^(k+1) = MLP^(k)( (1 + ε) · h_v^(k) + Σ_{u∈N(v)} h_u^(k) )
        # GINE extends this to incorporate edge features:
        #   h_v^(k+1) = MLP^(k)( (1 + ε) · h_v^(k) + Σ_{u∈N(v)} ReLU(h_u^(k) + e_uv) )
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            # The inner MLP for GINEConv: two linear layers with ReLU
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # --- Global pooling ---
        if pool == "mean":
            self.pool = global_mean_pool
        elif pool == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pool type: {pool!r}. Use 'mean' or 'sum'.")

        # --- MLP prediction head ---
        # Two linear layers: hidden_dim → hidden_dim → 1
        # The intermediate ReLU + Dropout add non-linearity and regularisation.
        # A single linear layer would be too restrictive (the pooled
        # representation needs non-linear transformation to predict the target).
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MoleculeGNN: {num_layers} layers, hidden={hidden_dim}, "
            f"pool={pool}, dropout={dropout}, task={task_type}, "
            f"params={n_params:,}"
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass: molecular graph → scalar prediction.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Must contain:
            - ``x``: atom feature matrix (num_atoms, atom_feat_dim)
            - ``edge_index``: COO edge indices (2, num_edges)
            - ``edge_attr``: bond feature matrix (num_edges, bond_feat_dim)
            - ``batch``: batch assignment vector (num_atoms,) — which
              atom belongs to which molecule in the batch. PyG's
              DataLoader creates this automatically.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 1). Raw logits for regression, or
            sigmoid probabilities for classification.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Step 1: Embed atoms and bonds into hidden space
        x = self.atom_encoder(x)           # (num_atoms, hidden_dim)
        edge_attr = self.bond_encoder(edge_attr)  # (num_edges, hidden_dim)

        # Step 2: Message passing — N rounds of neighbor aggregation
        # Each round:
        #   1. Each atom collects messages from neighbors (sum of
        #      neighbor features + bond features)
        #   2. Updates its own representation via the MLP
        #   3. BatchNorm stabilises training
        #   4. ReLU introduces non-linearity
        #   5. Dropout regularises
        #
        # Residual connections (x = x + conv(...)) help gradient flow
        # in deeper networks. Without them, stacking >3 layers often
        # hurts performance ("over-smoothing": all atoms converge to
        # similar representations).
        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = torch.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection

        # Step 3: Global pooling — aggregate atom representations
        # into a single vector per molecule.
        # `batch` tells the pool which atoms belong to which molecule
        # (e.g., [0,0,0,1,1,1,1,2,2] for 3 molecules with 3,4,2 atoms).
        graph_repr = self.pool(x, batch)   # (batch_size, hidden_dim)

        # Step 4: Prediction head
        out = self.head(graph_repr)        # (batch_size, 1)

        if self.task_type == "classification":
            out = torch.sigmoid(out)

        return out

    def get_config(self) -> dict:
        """Return model configuration as a serialisable dict.

        Useful for W&B logging and model checkpointing.
        """
        return {
            "model_type": "gnn",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "task_type": self.task_type,
            "dropout": self.dropout_rate,
            "pool": self.pool_name,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from molgate.data.featurizer import smiles_to_graph

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Build a small batch of molecular graphs ---
    molecules = [
        ("CCO", -0.77),            # ethanol
        ("c1ccccc1", -0.77),       # benzene
        ("CC(=O)Oc1ccccc1C(=O)O", -1.13),  # aspirin
        ("CC(=O)O", 0.17),         # acetic acid
    ]

    from torch_geometric.data import Batch

    graphs = []
    for smi, target in molecules:
        g = smiles_to_graph(smi, y=target)
        if g is not None:
            graphs.append(g)

    batch = Batch.from_data_list(graphs)

    print(f"Batch: {batch}")
    print(f"  num_graphs = {batch.num_graphs}")
    print(f"  x shape    = {batch.x.shape}")
    print(f"  edge_index = {batch.edge_index.shape}")
    print(f"  edge_attr  = {batch.edge_attr.shape}")
    print(f"  batch      = {batch.batch.shape}")

    # --- Instantiate model and run a forward pass ---
    model = MoleculeGNN(
        hidden_dim=128,
        num_layers=3,
        task_type="regression",
        dropout=0.1,
        pool="mean",
    )

    model.eval()
    with torch.no_grad():
        preds = model(batch)

    print(f"\nPredictions (untrained, random weights):")
    for (smi, target), pred in zip(molecules, preds.squeeze()):
        print(f"  {smi:30s}  target={target:+.3f}  pred={pred.item():+.3f}")

    print(f"\nModel config: {model.get_config()}")

    # --- Drop into IPython ---
    import IPython
    IPython.embed()
