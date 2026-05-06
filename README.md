# molgate

ADMET property prediction pipeline using graph neural networks. Designed as a filter step in virtual screening — compounds are first rejected by hard binary filters (hERG, DILI), then surviving compounds are scored on continuous ADMET properties.

A secondary goal is systematic analysis of how training set composition (split strategy, property distribution bias, molecular weight range) affects GNN generalization.

## Stack

| Component | Library |
|---|---|
| Featurization | DeepChem `MolGraphConvFeaturizer` |
| GNN | PyTorch Geometric (NNConv MPNN) |
| Datasets | TDC (Therapeutics Data Commons) |
| Experiment tracking | Weights & Biases |
| Molecular utilities | RDKit |
| Inference API | FastAPI |

## Properties covered

**Tier 1 — hard filters (binary):** hERG inhibition, DILI

**Tier 2 — soft scoring (regression):** solubility, Caco-2, CYP3A4/2D6/2C9, half-life, clearance

## Quick start

```bash
# Train solubility model on the fixed TDC benchmark split (scaffold, seed 42)
python scripts/train_property.py --property solubility --split scaffold --seed 42

# Run a bias study grid (48 runs: 2 splits × 8 bias variants × 3 seeds) — dry run first
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --dry-run
python scripts/run_bias_study.py configs/experiments/solubility_bias_study.yaml --no-wandb

# Evaluate all 22 TDC benchmark datasets across 5 seeds (leaderboard-format output)
python scripts/benchmark_eval.py --seeds 42 123 456 789 1337 --no-wandb
```

All bias experiments evaluate on the same fixed TDC test set, so biased-model performance is directly comparable to leaderboard numbers.

See `GUIDE.md` for full usage and `ARCHITECTURE.md` for design details.

## Tests

```bash
pytest tests/
```
