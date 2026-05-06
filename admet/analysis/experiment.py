"""ExperimentSpec dataclass and experiment grid expansion.

An experiment grid is defined in a YAML config (configs/experiments/*.yaml)
as the cartesian product of split_variants × bias_variants × seeds.
expand_experiment_grid() enumerates this product into a flat list of
ExperimentSpec objects that runner.py can execute independently.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf

from admet.analysis.bias import (
    AnyBiasConfig,
    ClusterBias,
    ClassImbalanceBias,
    MolWeightRangeBias,
    PropertyQuantileBias,
    ScaffoldSubsetBias,
)
from admet.model import ModelConfig
from admet.train import TrainingConfig


@dataclass
class ExperimentSpec:
    """Full specification for one training run in a bias study.

    Every field is serializable (used for W&B config and result JSON logging).
    The run_id property produces a deterministic, human-readable identifier
    that is unique within an experiment grid.

    data_source controls which loading function is used:
    - "benchmark": load_benchmark_split() — fixed test set, leaderboard-comparable.
      All bias experiments should use this. Default.
    - "single_pred": load_tdc_split() — dynamic splits, for exploration only.
    """

    property_name: str
    tdc_dataset_name: str
    task_type: Literal["regression", "classification"]
    split_method: Literal["random", "scaffold"]
    bias_config: AnyBiasConfig | None
    seed: int
    model_config: ModelConfig
    training_config: TrainingConfig
    wandb_project: str
    wandb_group: str
    results_dir: Path
    data_source: Literal["benchmark", "single_pred"] = "benchmark"

    @property
    def bias_type(self) -> str:
        return self.bias_config.type if self.bias_config is not None else "no_bias"

    @property
    def bias_fingerprint(self) -> str:
        """Short identifier distinguishing bias variants of the same type.

        For 'no_bias' returns 'no_bias'. For typed bias configs, appends
        a 6-char hash of the serialized params so two configs of the same
        type but different parameters get different run IDs.
        """
        if self.bias_config is None:
            return "no_bias"
        params = self.bias_config.model_dump(exclude={"type"})
        h = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:6]
        return f"{self.bias_config.type}_{h}"

    @property
    def run_id(self) -> str:
        """Deterministic, collision-free identifier.

        Format: {property}_{split}_{bias_fingerprint}_{seed}
        """
        return f"{self.property_name}_{self.split_method}_{self.bias_fingerprint}_{self.seed}"

    def to_wandb_config(self) -> dict:
        """Flat config dict suitable for wandb.init(config=...)."""
        return {
            "property": self.property_name,
            "tdc_dataset_name": self.tdc_dataset_name,
            "task_type": self.task_type,
            "data_source": self.data_source,
            "split_method": self.split_method,
            "bias_type": self.bias_type,
            "bias_params": (
                self.bias_config.model_dump(exclude={"type"})
                if self.bias_config is not None
                else {}
            ),
            "seed": self.seed,
            "model_hidden_dim": self.model_config.hidden_dim,
            "model_num_layers": self.model_config.num_layers,
            "model_dropout": self.model_config.dropout,
            "epochs": self.training_config.epochs,
            "lr": self.training_config.lr,
            "batch_size": self.training_config.batch_size,
            "patience": self.training_config.patience,
        }


def expand_experiment_grid(cfg: DictConfig) -> list[ExperimentSpec]:
    """Expand an experiment YAML config into a flat list of ExperimentSpecs.

    The grid is the cartesian product of split_variants × bias_variants × seeds.

    Args:
        cfg: OmegaConf DictConfig loaded from configs/experiments/*.yaml.

    Returns:
        Flat list of ExperimentSpecs, one per (split, bias, seed) combination.
    """
    import yaml as _yaml

    props_path = Path(__file__).parent.parent.parent / "configs" / "properties.yaml"
    with open(props_path) as f:
        props = _yaml.safe_load(f)["properties"]

    prop_name = cfg.property
    if prop_name not in props:
        raise ValueError(
            f"Property '{prop_name}' not found in properties.yaml. "
            f"Available: {list(props.keys())}"
        )
    prop_cfg = props[prop_name]
    tdc_name = prop_cfg["tdc_name"]
    task_type = prop_cfg["task_type"]

    # data_source defaults to "benchmark" when not set in YAML
    data_source: Literal["benchmark", "single_pred"] = getattr(
        cfg, "data_source", "benchmark"
    )

    model_cfg = ModelConfig(
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        task_type=task_type,
    )

    results_dir = Path(cfg.results_dir)
    checkpoint_dir = results_dir.parent / "checkpoints"

    specs: list[ExperimentSpec] = []
    for split, bias_raw, seed in itertools.product(
        list(cfg.split_variants),
        list(cfg.bias_variants),
        list(cfg.seeds),
    ):
        bias_config = _parse_bias_config(bias_raw)
        train_cfg = TrainingConfig(
            epochs=cfg.training.epochs,
            lr=cfg.training.lr,
            batch_size=cfg.training.batch_size,
            patience=cfg.training.patience,
            checkpoint_dir=checkpoint_dir / prop_name,
            seed=seed,
            wandb_log=True,
        )
        specs.append(
            ExperimentSpec(
                property_name=prop_name,
                tdc_dataset_name=tdc_name,
                task_type=task_type,
                split_method=split,
                bias_config=bias_config,
                seed=seed,
                model_config=model_cfg,
                training_config=train_cfg,
                wandb_project=cfg.wandb_project,
                wandb_group=cfg.wandb_group,
                results_dir=results_dir,
                data_source=data_source,
            )
        )

    return specs


def _parse_bias_config(raw: object) -> AnyBiasConfig | None:
    """Parse a bias variant entry from YAML (None/null → no bias)."""
    if raw is None:
        return None

    if hasattr(raw, "_metadata"):  # OmegaConf node
        raw = OmegaConf.to_container(raw, resolve=True)

    if isinstance(raw, dict):
        bias_type = raw.get("type")
        kwargs = {k: v for k, v in raw.items() if k != "type"}
        if bias_type == "property_quantile":
            return PropertyQuantileBias(**kwargs)
        if bias_type == "mw_range":
            return MolWeightRangeBias(**kwargs)
        if bias_type == "class_imbalance":
            return ClassImbalanceBias(**kwargs)
        if bias_type == "scaffold_subset":
            return ScaffoldSubsetBias(**kwargs)
        if bias_type == "cluster":
            return ClusterBias(**kwargs)
        raise ValueError(f"Unknown bias type: '{bias_type}'")

    raise ValueError(f"Cannot parse bias config from: {raw!r}")
