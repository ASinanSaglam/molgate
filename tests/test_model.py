"""Tests for admet/model.py."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch


@pytest.fixture
def molecule_batch(rdkit_featurizer, sample_smiles):
    graphs = [rdkit_featurizer.featurize(s) for s in sample_smiles[:4]]
    return Batch.from_data_list([g for g in graphs if g is not None])


def test_regression_output_shape(molecule_batch):
    from admet.model import ADMETModel, ModelConfig
    model = ADMETModel(ModelConfig(task_type="regression"))
    out = model(molecule_batch)
    n = molecule_batch.num_graphs
    assert out.shape == (n, 1), f"Expected ({n}, 1), got {out.shape}"


def test_classification_output_shape(molecule_batch):
    from admet.model import ADMETModel, ModelConfig
    model = ADMETModel(ModelConfig(task_type="classification"))
    out = model(molecule_batch)
    n = molecule_batch.num_graphs
    assert out.shape == (n, 1)


def test_predict_proba_classification(molecule_batch):
    from admet.model import ADMETModel, ModelConfig
    model = ADMETModel(ModelConfig(task_type="classification"))
    proba = model.predict_proba(molecule_batch)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_predict_proba_raises_for_regression(molecule_batch):
    from admet.model import ADMETModel, ModelConfig
    model = ADMETModel(ModelConfig(task_type="regression"))
    with pytest.raises(RuntimeError, match="predict_proba"):
        model.predict_proba(molecule_batch)


def test_model_config_roundtrip(tmp_path):
    """Checkpoint saves ModelConfig and it can reconstruct the model."""
    from admet.model import ADMETModel, ModelConfig
    from admet.train import load_checkpoint
    import torch

    cfg = ModelConfig(hidden_dim=64, num_layers=2, task_type="regression")
    model = ADMETModel(cfg)

    ckpt_path = tmp_path / "test.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": cfg,
        "training_config": None,
        "val_metric": 1.0,
        "val_metric_name": "val_loss",
        "tdc_dataset_name": "ESOL",
        "split_method": "scaffold",
        "epoch": 1,
    }, ckpt_path)

    loaded_model, ckpt = load_checkpoint(ckpt_path)
    assert ckpt["model_config"].hidden_dim == 64
    assert ckpt["tdc_dataset_name"] == "ESOL"
