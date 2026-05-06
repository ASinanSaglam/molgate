# ADMET Property Prediction Pipeline

## Project Overview

A modular ADMET property prediction tool built with graph neural networks, designed as a filter step in a larger virtual screening pipeline. The goal is a clean, production-quality codebase that demonstrates end-to-end ML engineering — data loading, featurization, model training, evaluation, and inference — not just a model.

This is a portfolio project but should be written as if it will be used in production. Prioritize clarity, modularity, and reproducibility over architectural complexity. A key focus for the project is also to do a analysis of the GNN based on different training sets. Make sure to discuss and include a structure and mechanisms to analyze the data, separate by different properties and retrain the GNN to see how different types of datasets impact the resulting model. 

---

## Stack

- **Featurization**: DeepChem `MolGraphConvFeaturizer` for atom and bond features
- **GNN Backend**: PyTorch Geometric
- **Datasets**: TDC (Therapeutics Data Commons) — standardized ADMET benchmarks with fixed splits
- **Experiment Tracking**: Weights & Biases
- **Molecular utilities**: RDKit
- **Inference API**: FastAPI (simple endpoint: accepts SMILES, returns predictions)

---

## Architecture

### Core Design

Separate model per ADMET property. Each model shares the same featurizer and GNN backbone architecture but has its own output head and trained weights. This keeps models independently debuggable and swappable.

Do not build a multi-task model in the first version.

### GNN Backbone

Start with a straightforward message passing network (MPNN or GCN). Keep it simple — the goal is a clean baseline with benchmarkable results, not a novel architecture. Use the same backbone class for all property models, parameterized by output type (regression vs classification).

### Output Heads

- **Regression** (e.g., solubility, permeability, clearance): single linear output, MSE loss
- **Classification** (e.g., hERG inhibition, DILI, BBB): single linear output + sigmoid, BCE loss

### Pipeline Filter Logic

The pipeline wrapper runs models in two tiers:

**Tier 1 — Hard filters (binary, sequential):**
Compounds failing these are immediately rejected. Order matters for efficiency — run cheapest/most-rejecting first.
1. hERG inhibition
2. DILI (drug-induced liver injury)

**Tier 2 — Soft scoring (regression, parallel):**
Surviving compounds are scored on continuous properties and ranked.
- Solubility
- Caco-2 permeability
- CYP3A4 inhibition
- CYP2D6 inhibition
- CYP2C9 inhibition
- Half-life
- Clearance

The pipeline wrapper returns: `passed` (bool), `hard_filter_reason` (str or None), and a `scores` dict with property values for Tier 2.

---

## Repo Structure

```
admet-pipeline/
├── CLAUDE.md                  # This file
├── README.md
├── requirements.txt
├── configs/
│   └── properties.yaml        # Per-property config: dataset name, task type, threshold (for Tier 1)
├── data/
│   └── (TDC datasets cached here, gitignored)
├── admet/
│   ├── __init__.py
│   ├── featurizer.py          # DeepChem featurizer wrapper
│   ├── dataset.py             # TDC data loading, train/val/test splits → PyG Data objects
│   ├── model.py               # GNN backbone + output head
│   ├── train.py               # Training loop, W&B logging, model checkpointing
│   ├── evaluate.py            # Metrics: AUROC for classification, RMSE/R² for regression
│   └── pipeline.py            # Two-tier filter wrapper
├── scripts/
│   ├── train_property.py      # CLI: train a single property model
│   └── run_pipeline.py        # CLI: run full ADMET filter on a SMILES list
├── api/
│   └── app.py                 # FastAPI app: POST /predict, accepts SMILES, returns scores
├── notebooks/
│   └── exploration.ipynb      # EDA, sanity checks, result visualization
└── tests/
    ├── test_featurizer.py
    ├── test_dataset.py
    └── test_pipeline.py
```

---

## Development Conventions

- One model, one property. Do not mix concerns.
- All dataset splits come from TDC — do not create custom splits.
- Results must be reproducible: seed everything (torch, numpy, random) and log the seed to W&B.
- Every model checkpoint saves: weights, config, val metric, TDC dataset name and split version.
- Type hints everywhere. No untyped function signatures.
- Docstrings on all public classes and functions.
- No Jupyter notebooks for training — notebooks are for exploration only.

---

## Build Order

Work in this sequence. Do not skip ahead.

1. `admet/featurizer.py` — wrap DeepChem featurizer, verify output shape on a test SMILES
2. `admet/dataset.py` — TDC loading for one property (start with ESOL/solubility), convert to PyG Data objects
3. `admet/model.py` — GNN backbone + regression head, verify forward pass
4. `admet/train.py` — training loop with W&B logging
5. `admet/evaluate.py` — metrics
6. Train and benchmark solubility as the first end-to-end test
7. Extend to classification (hERG) — add classification head, BCE loss, AUROC metric
8. Add remaining properties via config
9. `admet/pipeline.py` — two-tier filter wrapper
10. `api/app.py` — FastAPI inference endpoint
11. Tests

---

## Background Context

This project is being built by a computational chemist with a PhD and ~10 years of scientific software development experience, including 2.5 years at a biotech AI startup doing computational drug discovery (ABFE, virtual screening, structure prediction pipelines). The developer is highly familiar with RDKit and molecular simulation but is actively building ML engineering depth.

The ADMET pipeline is intended as a filter step in a larger virtual screening pipeline being developed in parallel. Keep interfaces clean and modular with that downstream integration in mind.