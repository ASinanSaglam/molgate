# End-to-End Usage Guide

This guide covers everything from installation through running bias studies and interpreting results.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .   # installs admet/ as a package
```

**Note:** PyTorch Geometric extras (`torch-scatter`, `torch-sparse`, etc.) must be installed separately and matched to your CUDA version. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). The pipeline falls back gracefully to CPU if CUDA extensions are unavailable.

DeepChem is the default featurizer. If it is not installed, the pipeline automatically falls back to an RDKit-only featurizer that produces the same tensor shapes (30-dim atom, 11-dim bond features) so the GNN backbone is compatible with both.

---

## Repository layout

```
molgate/
├── configs/
│   ├── properties.yaml              # canonical per-property config (do not edit during experiments)
│   └── experiments/
│       ├── solubility_bias_study.yaml
│       └── herg_bias_study.yaml
├── admet/
│   ├── featurizer.py                # molecular featurizer (DeepChem + RDKit fallback)
│   ├── dataset.py                   # TDC loading, bias application, PyG Dataset
│   ├── model.py                     # GNN backbone + output heads
│   ├── train.py                     # training loop, checkpointing, W&B
│   ├── evaluate.py                  # metrics (RMSE/R²/Spearman, AUROC/AUPRC)
│   ├── pipeline.py                  # two-tier filter wrapper (Step 9, not yet built)
│   └── analysis/
│       ├── bias.py                  # dataset bias transforms
│       ├── experiment.py            # ExperimentSpec dataclass + grid expansion
│       ├── runner.py                # run_experiment() + run_experiment_grid()
│       └── report.py                # load JSONs → summary DataFrames
├── scripts/
│   ├── train_property.py            # train one property model
│   ├── run_bias_study.py            # run a full experiment grid
│   └── run_pipeline.py              # run ADMET filter on a SMILES list (not yet built)
├── results/
│   ├── runs/                        # one JSON per training run (gitignored)
│   ├── tables/                      # summary CSVs committed to repo
│   └── figures/                     # committed figures from notebooks
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_split_comparison.ipynb
│   └── 03_bias_analysis.ipynb
└── tests/
```

---

## Training a single property model

```bash
python scripts/train_property.py \
  --property solubility \
  --split scaffold \
  --seed 42
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--property` | required | Property name (key in `properties.yaml`) |
| `--split` | property default | `random`, `scaffold`, or `temporal` |
| `--seed` | `42` | Random seed |
| `--data-source` | `benchmark` | `benchmark` (fixed TDC test set) or `single_pred` (dynamic splits) |
| `--epochs` | `100` | Max epochs |
| `--lr` | `1e-3` | Learning rate |
| `--batch-size` | `64` | Batch size |
| `--hidden-dim` | `128` | GNN hidden dimension |
| `--num-layers` | `3` | Number of message passing layers |
| `--no-wandb` | off | Disable W&B logging |
| `--wandb-project` | `molgate-production` | W&B project name |
| `--featurizer` | `auto` | `auto`, `deepchem`, or `rdkit` |

### Available properties

| Key | TDC dataset | Task | Tier |
|---|---|---|---|
| `solubility` | solubility_aqsoldb | regression | 2 |
| `herg` | herg | classification | 1 (hard filter) |
| `dili` | dili | classification | 1 (hard filter) |
| `caco2` | caco2_wang | regression | 2 |
| `cyp3a4` | cyp3a4_substrate_carbonmangels | classification | 2 |
| `cyp2d6` | cyp2d6_substrate_carbonmangels | classification | 2 |
| `cyp2c9` | cyp2c9_substrate_carbonmangels | classification | 2 |
| `halflife` | half_life_obach | regression | 2 |
| `clearance` | clearance_hepatocyte_az | regression | 2 |

### Output

Training logs to W&B and saves a checkpoint:

```
checkpoints/{property}/{tdc_name}_{split}_seed{seed}_ep{epoch}.pt
```

Each checkpoint contains:
- `model_state_dict` — model weights
- `model_config` — `ModelConfig` dataclass (reconstructs the architecture)
- `training_config` — `TrainingConfig` dataclass
- `val_metric` — best validation loss
- `tdc_dataset_name`, `split_method`, `epoch`

Reload a checkpoint:

```python
from admet.train import load_checkpoint

model, ckpt = load_checkpoint("checkpoints/solubility/solubility_aqsoldb_scaffold_seed42_ep67.pt")
# model is in eval mode
print(ckpt["tdc_dataset_name"], ckpt["val_metric"])
```

---

## Running a bias study

A bias study sweeps the cartesian product of **split strategies × bias configurations × random seeds** for one property, producing a JSON result per run and summary CSVs.

### Step 1: Review the experiment config

```yaml
# configs/experiments/solubility_bias_study.yaml
experiment_name: solubility_split_and_bias
property: solubility
data_source: benchmark          # fixed TDC test set — all runs comparable to leaderboard
wandb_project: molgate-bias-study
wandb_group: solubility_split_comparison

seeds: [42, 123, 456]

split_variants:
  - random
  - scaffold

bias_variants:
  - null                                     # baseline: no bias
  - type: property_quantile
    low_quantile: 0.0
    high_quantile: 0.33                      # train on low-solubility compounds only
  - type: property_quantile
    low_quantile: 0.67
    high_quantile: 1.0                       # train on high-solubility compounds only
  - type: mw_range
    max_mw: 300.0                            # fragment-like training set
  - type: mw_range
    min_mw: 500.0                            # larger molecules only
  - type: cluster
    cluster_ids: [0]                         # largest Butina cluster only
    butina_cutoff: 0.4
  - type: cluster
    cluster_ids: [0, 1, 2]                  # top 3 clusters
    butina_cutoff: 0.4
  - type: cluster
    cluster_ids: [0]
    butina_cutoff: 0.4
    invert: true                             # everything except the largest cluster
```

This grid = 2 splits × 8 bias variants × 3 seeds = **48 runs**. `data_source: benchmark` locks the test set to the official TDC fixed split so every run's test performance is directly comparable to leaderboard numbers.

### Step 2: Dry run

Always do a dry run first to inspect the expanded grid before committing compute:

```bash
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --dry-run
```

Output:
```
Experiment grid: 48 runs (3 seeds × 2 splits × 8 bias variants)

Dry run — 48 specs (not executing):
  1. solubility_random_no_bias_42  bias=no_bias
  2. solubility_random_no_bias_123  bias=no_bias
  ...
 16. solubility_random_cluster_880953_42  bias={"type":"cluster","cluster_ids":[0],...}
  ...
 25. solubility_scaffold_no_bias_42  bias=no_bias
  ...
```

Bias variants of the same type but different parameters get a 6-character hash appended to the `run_id` (e.g., `cluster_880953` vs `cluster_f0a7f3`) so result JSONs never collide.

### Step 3: Run the grid

```bash
# Sequential (safe on single GPU)
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml

# Parallel (set --n-parallel to number of available GPUs or CPU cores)
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --n-parallel 4

# Without W&B (for testing)
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --no-wandb

# Run only the first 2 specs (smoke test)
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --limit 2 --no-wandb
```

Each completed run writes a JSON to `results/runs/`:

```json
{
  "run_id": "solubility_scaffold_property_quantile_28f058_42",
  "property": "solubility",
  "tdc_dataset_name": "solubility_aqsoldb",
  "data_source": "benchmark",
  "split_method": "scaffold",
  "seed": 42,
  "bias_type": "property_quantile",
  "bias_params": {"low_quantile": 0.0, "high_quantile": 0.33},
  "train_size": 2279,
  "val_size": 998,
  "test_size": 1997,
  "train_mw_mean": 213.4,
  "train_y_mean": -4.82,
  "train_y_std": 0.87,
  "test_rmse": 2.14,
  "test_mae": 1.61,
  "test_r2": 0.31,
  "test_spearman": 0.54,
  "training_seconds": 145.3
}
```

After the grid finishes, summary CSVs are written automatically to `results/tables/`.

### Step 4: Analyze results

Load all results and compute summary tables in Python or from a notebook:

```python
from pathlib import Path
from admet.analysis.report import load_results, make_split_comparison_table, make_bias_sensitivity_table

results_dir = Path("results/runs")
df = load_results(results_dir)

# Effect of split strategy (baseline runs only, aggregated over seeds)
split_table = make_split_comparison_table(df)
print(split_table)
# Split:          random          scaffold
# Stat:           mean    std     mean    std
# solubility      1.32    0.04    1.41    0.06

# Effect of training set bias (delta from no-bias baseline)
bias_table = make_bias_sensitivity_table(df)
print(bias_table)
# bias_type:            mw_range  property_quantile
# solubility              +0.18         +0.52
```

Positive delta = metric improved vs baseline (for AUROC, R²); sign is consistent.

---

## Defining a new bias experiment

To add a new property or a new bias configuration, create a new YAML in `configs/experiments/`:

```yaml
experiment_name: herg_split_and_bias
property: herg
wandb_project: molgate-bias-study
wandb_group: herg_split_comparison

seeds: [42, 123, 456]

split_variants:
  - random
  - scaffold

bias_variants:
  - null
  - type: class_imbalance
    positive_fraction: 0.1
    strategy: undersample_majority    # or oversample_minority
  - type: class_imbalance
    positive_fraction: 0.5
    strategy: undersample_majority
  - type: mw_range
    max_mw: 400.0

model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.1

training:
  epochs: 100
  lr: 0.001
  batch_size: 64
  patience: 15

results_dir: results/runs
```

### Available bias types

**`property_quantile`** — keep only molecules in a Y-value quantile band (regression).
- `low_quantile`: float [0, 1]
- `high_quantile`: float [0, 1]

**`mw_range`** — filter by molecular weight range.
- `min_mw`: float or null
- `max_mw`: float or null

**`class_imbalance`** — adjust positive/negative ratio in training (classification only).
- `positive_fraction`: float — target fraction of positive labels
- `strategy`: `undersample_majority` or `oversample_minority`
- `seed`: int (default 42)

**`scaffold_subset`** — restrict training to molecules with specific Murcko scaffolds.
- `scaffold_smiles`: list of SMILES strings
- `invert`: bool — if true, *exclude* these scaffolds instead

**`cluster`** — group molecules by chemical similarity using Butina clustering on Morgan fingerprints, then keep or exclude specified cluster indices.
- `cluster_ids`: list of ints — which clusters to keep (clusters are sorted largest-first, so `[0]` is the largest chemical series)
- `butina_cutoff`: float (default `0.4`) — Tanimoto distance threshold; smaller → more, tighter clusters
- `fingerprint_radius`: int (default `2`) — Morgan fingerprint radius
- `fingerprint_bits`: int (default `2048`) — fingerprint bit vector length
- `invert`: bool (default `false`) — if true, keep molecules *not* in the specified clusters

Clusters are computed at bias-application time from the training DataFrame, so they adapt to whatever split is active. Molecules whose SMILES cannot be parsed are appended as singleton clusters at the end.

```yaml
# Example: train only on the single most-populated chemical series
- type: cluster
  cluster_ids: [0]
  butina_cutoff: 0.4

# Example: hold out the largest cluster to test cross-series generalization
- type: cluster
  cluster_ids: [0]
  butina_cutoff: 0.4
  invert: true
```

---

## TDC leaderboard evaluation

`scripts/benchmark_eval.py` trains models across all 22 TDC ADMET benchmark datasets and produces output in the format required for official leaderboard submission.

```bash
# Full leaderboard run — all 22 datasets, 5 seeds (TDC minimum)
python scripts/benchmark_eval.py --seeds 42 123 456 789 1337

# Quick validation — two datasets, 2 seeds, no W&B
python scripts/benchmark_eval.py --datasets solubility_aqsoldb herg --seeds 42 123 --no-wandb
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--seeds` | `42 123 456 789 1337` | Random seeds (TDC requires ≥5 for official submission) |
| `--datasets` | all 22 | Subset of benchmark dataset names |
| `--split` | `scaffold` | Train/val partition strategy (test is always the fixed benchmark set) |
| `--epochs` | `100` | Max training epochs |
| `--hidden-dim` | `128` | GNN hidden dimension |
| `--num-layers` | `3` | Number of message passing layers |
| `--batch-size` | `64` | Batch size |
| `--patience` | `15` | Early stopping patience |
| `--no-wandb` | off | Disable W&B logging |
| `--output-dir` | `results/benchmark` | Directory for result JSONs |

### Output

The script writes two files to `--output-dir`:

- **`leaderboard_results.json`** — TDC-format output from `bg.evaluate_many()`:
  ```json
  {"solubility_aqsoldb": [1.21, 0.04], "herg": [0.89, 0.02], ...}
  ```
  (Each value is `[mean, std]` across seeds, computed by TDC's own evaluation code.)

- **`per_dataset_summary.json`** — our own per-metric aggregation:
  ```json
  {"solubility_aqsoldb": {"rmse": {"mean": 1.21, "std": 0.04}, "r2": {...}}, ...}
  ```

Because `data_source: benchmark` is the default everywhere, **all bias study runs evaluate on the same fixed test set as the leaderboard**. You can directly compare, e.g., a `cluster`-biased model's `test_rmse` against the leaderboard number.

---

## Using the featurizer and dataset API directly

```python
from pathlib import Path
from admet.featurizer import get_featurizer
from admet.dataset import load_benchmark_split, compute_split_statistics
from admet.analysis.bias import PropertyQuantileBias, ClusterBias

featurizer = get_featurizer("auto")   # DeepChem if available, else RDKit

# Cluster bias: train on largest chemical series only
bias = ClusterBias(cluster_ids=[0], butina_cutoff=0.4)

# load_benchmark_split uses the fixed TDC test set (leaderboard-compatible)
train_ds, val_ds, test_ds = load_benchmark_split(
    benchmark_name="solubility_aqsoldb",
    split_type="scaffold",
    seed=42,
    data_dir=Path("data"),
    featurizer=featurizer,
    task_type="regression",
    train_bias=bias,
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
print(f"Node features: {train_ds.num_node_features}")  # 30
print(f"Edge features: {train_ds.num_edge_features}")  # 11

# Dataset statistics (for logging or inspection)
train_stats = compute_split_statistics(train_ds._df, "regression")
print(train_stats)
# {'n': 2279.0, 'mw_mean': 213.4, ..., 'y_mean': -4.82, 'y_std': 0.87}
```

**Important:** bias is applied to the training split only. `val_ds` and `test_ds` always use the full, unmodified TDC fixed splits.

For exploration without leaderboard constraints, `load_tdc_split()` provides dynamic splits per seed via `tdc.single_pred`.

### Aligned predictions for leaderboard submission

When molecules fail featurization (invalid SMILES), the dataset is shorter than the original DataFrame. `predict_aligned()` fills those gaps so the prediction array matches the test DataFrame length exactly — required for `bg.evaluate_many()`:

```python
from admet.evaluate import predict_aligned

# Returns np.ndarray of shape (test_ds._n_total,)
# Featurization failures are filled with the training mean (regression)
# or 0.5 (classification)
aligned_preds = predict_aligned(model, test_ds)
```

---

## Training and evaluating programmatically

```python
import torch
from admet.model import ADMETModel, ModelConfig
from admet.train import TrainingConfig, train
from admet.evaluate import evaluate

model_cfg = ModelConfig(
    hidden_dim=128,
    num_layers=3,
    dropout=0.1,
    num_node_features=train_ds.num_node_features,
    num_edge_features=train_ds.num_edge_features,
    task_type="regression",
)
model = ADMETModel(model_cfg)

train_cfg = TrainingConfig(
    epochs=100,
    lr=1e-3,
    batch_size=64,
    patience=15,
    checkpoint_dir=Path("checkpoints/solubility"),
    seed=42,
    wandb_log=False,   # set True to enable W&B
)

ckpt_path = train(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    cfg=train_cfg,
    split_stats={
        "train": compute_split_statistics(train_ds._df, "regression"),
        "val":   compute_split_statistics(val_ds._df, "regression"),
        "test":  compute_split_statistics(test_ds._df, "regression"),
    },
    tdc_dataset_name="solubility_aqsoldb",
    split_method="scaffold",
    wandb_run=None,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
metrics = evaluate(model, test_ds)
print(metrics)
# {'rmse': 1.21, 'r2': 0.83, 'spearman': 0.91}
```

---

## W&B integration

All runs log to Weights & Biases when `--no-wandb` is not set. Two projects are used:

- **`molgate-production`** — canonical models trained with default splits; used for pipeline deployment
- **`molgate-bias-study`** — all bias analysis runs; grouped by `{property}_{experiment_name}`

Config logged per run includes: property, TDC dataset name, split method, seed, bias type and params, dataset statistics (MW distribution, Y distribution, class balance for classification), and all model/training hyperparameters.

This makes it straightforward to use the W&B comparison UI to isolate the effect of any single factor across runs.

---

## Running tests

```bash
pytest tests/ -v
```

The test suite covers featurizer output shapes, dataset loading (without TDC network access), bias transforms (all five types including `ClusterBias`), model forward pass, checkpoint round-trips, split validation logic, and aligned prediction alignment. 40 tests, ~5 seconds.

To run a specific module:

```bash
pytest tests/test_bias.py -v
pytest tests/test_featurizer.py -v
```

---

## Adding a new property

1. Add a row to `configs/properties.yaml`:

```yaml
  bbb:
    tdc_name: bbb_martins
    task_type: classification
    metric: auroc
    tier: 2
    default_split: scaffold
```

2. Add it to `SUPPORTED_SPLITS` in `admet/dataset.py`:

```python
SUPPORTED_SPLITS: dict[str, list[SplitMethod]] = {
    ...
    "bbb_martins": ["random", "scaffold"],
}
```

3. Train it:

```bash
python scripts/train_property.py --property bbb --split scaffold --seed 42
```

4. Optionally create `configs/experiments/bbb_bias_study.yaml` and run a bias grid.

---

## Reproducibility

Every run seeds `torch`, `numpy`, and `random` from the `seed` field in `TrainingConfig` before any data loading or model initialization. The seed is logged to W&B and saved in the checkpoint. Given the same TDC dataset version, split method, seed, and model config, training is deterministic on the same hardware.

TDC datasets are cached to `data/` after first download. The `data/` directory is gitignored; pin TDC version in `requirements.txt` for reproducibility across machines.
