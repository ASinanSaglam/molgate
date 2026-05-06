"""GNN backbone and output heads for ADMET property prediction.

Architecture: NNConv-based message passing network (MPNN) with mean pooling.
Edge features are used to parameterize the message weight matrix via a small
MLP — analogous to using ligand-field interactions to modulate force constants
in a force field, but learned end-to-end.

All models share the same MPNNBackbone class; task-specific behavior (regression
vs classification, loss function) is isolated to the output head in ADMETModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import NNConv, global_mean_pool


@dataclass
class ModelConfig:
    """Configuration for the GNN backbone and output head.

    Args:
        hidden_dim: Width of all hidden layers.
        num_layers: Number of NNConv message passing layers.
        dropout: Dropout rate applied after each conv layer (except the last).
        num_node_features: Atom feature dimension (30 for MolGraphConvFeaturizer).
        num_edge_features: Bond feature dimension (11 for MolGraphConvFeaturizer).
        task_type: "regression" or "classification". Determines the output head.
            Classification head outputs a raw logit — apply BCEWithLogitsLoss
            externally, not sigmoid, for numerical stability.
    """

    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    num_node_features: int = 30
    num_edge_features: int = 11
    task_type: Literal["regression", "classification"] = "regression"


class MPNNBackbone(nn.Module):
    """NNConv message passing network producing graph-level embeddings.

    Each layer uses an edge MLP to parameterize the message weight matrix:
        edge_mlp: num_edge_features → hidden_dim → in_dim * out_dim
    This allows bond type, conjugation, and stereo information to modulate
    how atom features are aggregated across each bond.

    The first layer projects from num_node_features to hidden_dim; subsequent
    layers operate in hidden_dim × hidden_dim space. A mean pool over all
    node embeddings produces the final graph-level vector.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        e = cfg.num_edge_features

        convs: list[nn.Module] = []
        for i in range(cfg.num_layers):
            in_dim = cfg.num_node_features if i == 0 else h
            # Edge MLP: e → h → in_dim * h (parameterizes in_dim × h weight matrix)
            edge_mlp = nn.Sequential(
                nn.Linear(e, h),
                nn.ReLU(),
                nn.Linear(h, in_dim * h),
            )
            convs.append(NNConv(in_dim, h, edge_mlp, aggr="mean"))

        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Compute graph-level embeddings.

        Args:
            x: Node features (N_total, num_node_features).
            edge_index: Edge connectivity (2, E_total).
            edge_attr: Edge features (E_total, num_edge_features).
            batch: Batch vector mapping each node to its graph (N_total,).

        Returns:
            Graph embeddings (B, hidden_dim).
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.activation(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        return global_mean_pool(x, batch)


class ADMETModel(nn.Module):
    """Full ADMET property model: MPNNBackbone + task-specific output head.

    The output head is a single linear layer projecting from hidden_dim to 1.
    For regression: output is the raw predicted value.
    For classification: output is a raw logit. Use BCEWithLogitsLoss during
    training and apply sigmoid at inference time.

    This design (no sigmoid in forward) is numerically more stable than
    applying sigmoid before BCE loss.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = MPNNBackbone(cfg)
        self.head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, data: Data) -> Tensor:
        """Run a forward pass on a batched PyG Data object.

        Args:
            data: Batched PyG Data with x, edge_index, edge_attr, batch fields.

        Returns:
            Predictions of shape (B, 1). Raw logit for classification.
        """
        emb = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        return self.head(emb)

    def predict_proba(self, data: Data) -> Tensor:
        """Classification-only: return sigmoid-transformed probabilities.

        Do not use during training (numerical instability vs BCEWithLogitsLoss).
        """
        if self.cfg.task_type != "classification":
            raise RuntimeError("predict_proba is only valid for classification models.")
        return torch.sigmoid(self.forward(data))
