# Architecture

## Overview

The pipeline has three layers:

1. **Core ML pipeline** — featurizer → dataset → model → train → evaluate
2. **Analysis layer** — bias transforms, experiment grids, result aggregation
3. **Interface layer** — CLI scripts, FastAPI endpoint (planned), notebooks

Each property has its own independently trained model. There is no multi-task learning. This makes models independently debuggable, swappable, and benchmarkable.

---

## Featurization

**File:** `admet/featurizer.py`

Molecules are featurized using DeepChem's `MolGraphConvFeaturizer`, which produces fixed-dimension atom and bond feature vectors:

| Feature vector | Dimension | Source |
|---|---|---|
| Atom (node) features | 30 | Atom type, chirality, formal charge, H count, hybridization, aromaticity |
| Bond (edge) features | 11 | Bond type, conjugation, ring membership, stereo |

The featurizer is wrapped behind a `MolFeaturizer` protocol so the GNN backbone never sees DeepChem directly. An `RDKitFeaturizer` fallback produces the same-shaped tensors when DeepChem is unavailable, maintaining GNN compatibility.

`get_featurizer("auto")` tries DeepChem first, falls back to RDKit.

---

## Dataset loading

**File:** `admet/dataset.py`

All data comes from TDC (Therapeutics Data Commons). TDC provides:
- Standardized ADMET benchmarks
- Fixed scaffold/random splits via `data.get_split(method=..., seed=...)`
- DataFrame format: `Drug` (SMILES) + `Y` (label)

Two loading functions are provided depending on use case:

**`load_benchmark_split()`** — canonical path. Uses the fixed TDC benchmark test set via `admet_group`, making all runs directly comparable to leaderboard numbers. The scaffold/random split axis applies only to the `train_val` pool; the test set is always the same regardless of seed.

**`load_tdc_split()`** — exploration path. Uses `tdc.single_pred` with dynamic splits per seed. Test set varies with seed; not leaderboard-comparable.

```
load_benchmark_split():
    admet_group.get(name)
        │
        ├─ test_df  (fixed — same for all seeds and splits)
        └─ train_val_df
               │
               ▼ bg.get_train_valid_split(split_type, seed)
           train_df / val_df
               │
               ▼ apply_bias(train_df)  ← bias applied HERE, train only
           train_df (biased)
               │
               ▼ featurize each SMILES
ADMETDataset (PyG Dataset)  × 3 (train / val / test)
```

Invalid SMILES are skipped with a warning; the dataset index is compacted. The raw TDC DataFrame is retained as `._df` on each dataset for statistics computation. `_valid_positions` tracks which original DataFrame rows featurized successfully; `_n_total` is the original row count before any skips.

`SUPPORTED_SPLITS` is a dict mapping dataset names to valid split methods. Requesting an unsupported method (e.g., temporal split on a dataset without timestamps) raises `ValueError` with a clear message.

---

## GNN backbone

**File:** `admet/model.py`

The backbone is an NNConv-based MPNN (message passing neural network). NNConv uses the bond feature vector to parameterize the message weight matrix at each layer, allowing bond type, conjugation, and stereo to modulate how atom representations are aggregated.

```
Input: atom features (N, 30), bond features (E, 11)
  │
  ▼ NNConv layer 1: (N, 30) → (N, 128)
    edge MLP: 11 → 128 → 30×128 (parameterizes weight matrix)
  │
  ▼ NNConv layer 2: (N, 128) → (N, 128)
  │
  ▼ NNConv layer 3: (N, 128) → (N, 128)
  │
  ▼ mean pooling over nodes → (B, 128)
  │
  ▼ linear → (B, 1)
```

**Output heads:**

| Task | Head | Loss |
|---|---|---|
| Regression | Linear → scalar | MSELoss |
| Classification | Linear → logit (no sigmoid) | BCEWithLogitsLoss |

Classification heads output raw logits. Sigmoid is applied at inference time only (`predict_proba()`), not during training — this is numerically more stable than applying sigmoid before BCE loss.

**Parameter count** at default settings (hidden_dim=128, 3 layers): ~4.8M parameters. The majority are in the edge MLPs (each maps 11-dim bond features to a 128×128 weight matrix: 11 → 128 → 16,384 weights per layer).

---

## Training loop

**File:** `admet/train.py`

Standard supervised training with:
- **Optimizer:** Adam
- **LR scheduler:** ReduceLROnPlateau (halves LR when val loss plateaus, patience/2 window)
- **Early stopping:** stops when val loss does not improve for `patience` epochs
- **Seeding:** `torch`, `numpy`, and `random` seeded at the start of every run

W&B receives config (dataset stats, hyperparams, bias params) at init, then per-epoch `train_loss` and `val_loss`, and final test metrics.

Checkpoints are saved on every val improvement:

```python
{
    "model_state_dict": ...,
    "model_config": ModelConfig(...),   # sufficient to reconstruct the architecture
    "training_config": TrainingConfig(...),
    "val_metric": float,
    "tdc_dataset_name": str,
    "split_method": str,
    "epoch": int,
}
```

---

## Evaluation metrics

**File:** `admet/evaluate.py`

| Task | Metric | Notes |
|---|---|---|
| Regression | RMSE | Primary; lower is better |
| Regression | MAE | Used by several TDC datasets (e.g., caco2, solubility) as the leaderboard metric |
| Regression | R² | Coefficient of determination |
| Regression | Spearman ρ | Rank correlation; used for half-life and clearance where ordinal ranking matters more than absolute error |
| Classification | AUROC | Primary; threshold-free |
| Classification | AUPRC | Important when class imbalance is high |
| Classification | Balanced accuracy | At sigmoid > 0.5 threshold |

`predict()` returns `(y_true, y_pred)` as numpy arrays. For classification, `y_pred` contains sigmoid probabilities. The `evaluate()` convenience function runs inference and returns the full metrics dict.

`predict_aligned()` is a companion function required for TDC leaderboard submission. Because `ADMETDataset` silently skips invalid SMILES, the dataset may be shorter than the original DataFrame. `predict_aligned()` maps predictions back to the original row positions, filling featurization failures with the training mean (regression) or 0.5 (classification), so the returned array has exactly `dataset._n_total` elements and can be passed directly to `bg.evaluate_many()`.

---

## Analysis layer

**Files:** `admet/analysis/`

The analysis subpackage is a dependency of nothing in the core pipeline — it imports from `admet.dataset`, `admet.train`, and `admet.evaluate`, but the reverse is never true.

### Bias transforms (`bias.py`)

All bias transforms are pure functions on TDC DataFrames:

```
apply_bias(df, bias_config, rng) → new DataFrame
```

Five transform types, implemented as pydantic v2 models with a discriminated union on the `type` field:

| Type | What it does | Primary use |
|---|---|---|
| `property_quantile` | Keep molecules in a Y-value quantile band | Study effect of training on only one end of the property range |
| `mw_range` | Filter by molecular weight | Study chemical space restriction (fragments vs leads) |
| `class_imbalance` | Adjust positive/negative ratio | Study effect of class imbalance on classification models |
| `scaffold_subset` | Keep/exclude by Murcko scaffold list | Study scaffold generalization vs memorization |
| `cluster` | Keep/exclude by Butina cluster index | Study chemical-series bias — train on one series, test on all |

**ClusterBias implementation:**
Clustering is computed at bias-application time (never cached) from the active training DataFrame:

1. Compute Morgan fingerprints (radius 2, 2048 bits) for all parseable SMILES
2. Build the upper-triangle Tanimoto distance matrix (O(N²) — tractable for TDC-scale datasets up to ~7 000 molecules)
3. Run RDKit `Butina.ClusterData` with the specified `butina_cutoff` as a Tanimoto distance threshold
4. Sort clusters largest-first; map fingerprint-space indices back to DataFrame positions
5. Molecules that failed to parse are appended as singletons at the end

`cluster_ids` indexes into this sorted list (`[0]` is always the largest chemical series). `invert=True` returns the complement (all molecules not in the specified clusters). `get_cluster_assignments()` exposes the raw cluster list for EDA.

Bias is applied to the **training split only**. Val and test splits are never modified. This is enforced in `load_benchmark_split()` and `load_tdc_split()` — the bias parameter is named `train_bias` and is only passed to the training DataFrame.

### Experiment grid (`experiment.py`)

An `ExperimentSpec` is a complete, serializable specification for one training run:

```
property × TDC name × task type × split method × bias config × seed × model config × training config
```

`expand_experiment_grid(cfg)` takes an OmegaConf DictConfig from a YAML and produces the cartesian product:

```
split_variants × bias_variants × seeds → list[ExperimentSpec]
```

Each `ExperimentSpec` has a deterministic `run_id` that includes a 6-character MD5 hash of the bias parameters to avoid collisions between different instances of the same bias type (e.g., two `property_quantile` configs with different quantile bounds, or two `cluster` configs with different `cluster_ids`). `ExperimentSpec` also carries a `data_source` field (`"benchmark"` or `"single_pred"`) that the runner uses to route to the correct loading function.

### Runner (`runner.py`)

`run_experiment(spec)` executes one complete training run:

1. Route to `load_benchmark_split()` or `load_tdc_split()` based on `spec.data_source`
2. Compute split statistics
3. Init W&B run
4. Train model
5. Evaluate on test set; collect `predict_aligned()` output
6. Write result JSON to `results/runs/{run_id}.json`
7. Return result dict

`run_experiment_grid(specs, n_parallel=1)` runs a list of specs. With `n_parallel > 1`, uses `ProcessPoolExecutor` (not threads) for GIL-free execution with isolated GPU memory per worker. The default is sequential (`n_parallel=1`) to avoid OOM on single-GPU machines.

### Report (`report.py`)

Loads all result JSONs and produces two summary DataFrames:

**Split comparison table** — effect of split strategy on held-out performance (baseline runs only, aggregated over seeds):

```
              random          scaffold
              mean    std     mean    std
solubility    1.32    0.04    1.41    0.06
```

**Bias sensitivity table** — mean delta from the no-bias baseline for each bias type:

```
              mw_range   property_quantile
solubility    +0.18      +0.52
```

Positive = metric degraded vs baseline (for RMSE, higher is worse; table signs are raw delta, so positive RMSE delta = degradation). For AUROC the sign convention reverses.

---

## Configuration schema

### `configs/properties.yaml`

The canonical, stable property registry. Never modified during experiments.

```yaml
properties:
  {property_key}:
    tdc_name: str         # TDC dataset name (case-sensitive, lowercase)
    task_type: str        # "regression" or "classification"
    metric: str           # primary metric name
    tier: int             # 1 = hard filter, 2 = soft score
    threshold: float      # classification only: sigmoid threshold for Tier 1 rejection
    default_split: str    # "random" or "scaffold"
```

### `configs/experiments/*.yaml`

One file per experiment grid. Referenced only by `run_bias_study.py`.

```yaml
experiment_name: str
property: str               # must match a key in properties.yaml
data_source: str            # "benchmark" (default) or "single_pred"
wandb_project: str
wandb_group: str
seeds: list[int]
split_variants: list[str]   # ["random", "scaffold"]
bias_variants: list         # null or bias config dicts

model:
  hidden_dim: int
  num_layers: int
  dropout: float

training:
  epochs: int
  lr: float
  batch_size: int
  patience: int

results_dir: str            # e.g. "results/runs"
```

`data_source: benchmark` (the default) locks all runs to the fixed TDC test set, making every result directly comparable to the official leaderboard. `data_source: single_pred` allows dynamic seed-varying splits for exploratory work.

---

## Dependency graph

```
featurizer.py
    ↑
dataset.py ←── analysis/bias.py
    ↑
model.py
    ↑
train.py ←── analysis/experiment.py
    ↑               ↑
evaluate.py    analysis/runner.py
    ↑               ↑
    │         analysis/report.py
    │               ↑
    │         scripts/run_bias_study.py
    │               ↑
    │         notebooks/02_split_comparison.ipynb
    │         notebooks/03_bias_analysis.ipynb
    │
scripts/benchmark_eval.py   (imports dataset + evaluate directly, no analysis layer)
```

`analysis/` imports from the core pipeline. The core pipeline never imports from `analysis/`. This direction is enforced: adding an import from `admet.analysis` in `admet/train.py` would create a cycle.

`benchmark_eval.py` bypasses the analysis layer entirely — it calls `load_benchmark_split()`, `train()`, and `predict_aligned()` directly and assembles the `bg.evaluate_many()` submission dict itself.

---

## Two-tier pipeline (planned)

`admet/pipeline.py` (Step 9) will wrap all trained models in a two-stage filter:

**Tier 1 — hard filters, run sequentially:**
1. hERG — sigmoid > 0.5 → reject ("hERG inhibitor")
2. DILI — sigmoid > 0.5 → reject ("hepatotoxic")

Compounds failing Tier 1 are immediately returned with `passed=False` and a `hard_filter_reason` string. No further inference runs.

**Tier 2 — soft scoring, run in parallel:**
Compounds passing Tier 1 are scored on all regression properties simultaneously. Returns a `scores` dict.

Pipeline return type:
```python
{
    "passed": bool,
    "hard_filter_reason": str | None,
    "scores": {
        "solubility": float,
        "caco2": float,
        ...
    }
}
```

---

## FastAPI endpoint (planned)

`api/app.py` (Step 10) will expose a `POST /predict` endpoint:

```http
POST /predict
Content-Type: application/json

{"smiles": ["CCO", "c1ccccc1"]}
```

Response:
```json
[
  {"smiles": "CCO", "passed": true, "hard_filter_reason": null, "scores": {...}},
  {"smiles": "c1ccccc1", "passed": false, "hard_filter_reason": "hERG inhibitor", "scores": null}
]
```
