# Molecule Property Prediction with Prefect + W&B

## What we're building

A portfolio-quality repository that answers: **"How does dataset composition affect molecular property prediction models?"** It combines:
- Thorough data analysis (evolved from `molecule_property/`)
- Model training pipelines (architectural patterns from `prowhiz`)
- Intentional dataset bias experiments
- Prefect orchestration + W&B experiment tracking
- GitHub-hosted notebooks with narrative interpretations

---

## Repository structure

```
molgate/
├── src/molgate/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py          # TDC dataset loading, canonical SMILES, dedup
│   │   ├── featurizer.py       # Molecular featurization (descriptors, fingerprints, graph)
│   │   ├── splits.py           # Random, scaffold, stratified splitting
│   │   ├── bias.py             # Intentional bias generators
│   │   └── registry.py         # Dataset registry (name → task type, metric, etc.)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py         # Fingerprint + LightGBM / XGBoost
│   │   ├── gnn.py              # GNN model (MPNN or GCN via PyG)
│   │   ├── ensemble.py         # Stacking, blending, voting ensembles
│   │   └── factory.py          # Model factory from config (incl. ensembles)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop (GNN)
│   │   ├── metrics.py          # AUROC, RMSE, R², MAE, etc.
│   │   └── evaluate.py         # Evaluation + per-sample predictions
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── descriptors.py      # RDKit descriptor computation
│   │   ├── scaffolds.py        # Scaffold analysis utilities
│   │   ├── cliffs.py           # Activity cliff detection
│   │   └── bias_diagnostics.py # Bias characterization (distribution shift, diversity)
│   ├── serve/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI prediction endpoint
│   │   └── predict.py          # Model loading + featurization + inference
│   ├── cli.py                  # CLI entrypoint (molgate predict "CCO")
│   └── flows/
│       ├── __init__.py
│       ├── eda_flow.py         # Prefect flow: full EDA for a dataset
│       ├── train_flow.py       # Prefect flow: train single model on single dataset variant
│       ├── bias_experiment.py  # Prefect flow: run full bias experiment matrix
│       └── compare_flow.py     # Prefect flow: aggregate results, generate comparison artifacts
├── notebooks/
│   ├── 01_data_overview.ipynb      # Dataset landscape, all 22 ADMET properties
│   ├── 02_solubility_deep_dive.ipynb  # Solubility EDA + baseline
│   ├── 03_bias_design.ipynb        # Explain bias generation strategies, visualize biased datasets
│   ├── 04_model_comparison.ipynb   # Results across models × datasets (pulls from W&B)
│   ├── 05_bias_impact.ipynb        # Core finding: how bias affects model performance
│   ├── 06_ensemble_lab.ipynb       # Ensemble exploration: stacking, blending, voting
│   └── 07_conclusions.ipynb        # Summary, lessons learned, recommendations
├── configs/
│   ├── datasets.yaml           # Dataset-specific configs (which properties, metrics)
│   ├── models.yaml             # Model hyperparameter configs
│   ├── bias_experiments.yaml   # Bias experiment definitions
│   └── wandb.yaml              # W&B project/entity settings
├── tests/
│   ├── conftest.py
│   ├── test_loaders.py
│   ├── test_featurizer.py
│   ├── test_splits.py
│   ├── test_bias.py
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_flows.py
├── pyproject.toml
├── Makefile
├── README.md                   # Project overview, results summary, how to reproduce
└── .github/workflows/ci.yml
```

---

## Phase 0: Project scaffolding

**Goal**: Empty but functional project skeleton.

- [x] 0.1 — Create repository with `pyproject.toml` (hatchling, dependencies: torch, torch-geometric, rdkit, lightgbm, xgboost, pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, prefect, wandb, tdc, optuna)
- [x] 0.2 — Create `src/molgate/__init__.py` with version
- [x] 0.3 — Create empty module directories with `__init__.py` files
- [x] 0.4 — Create `Makefile` with targets: `install`, `test`, `lint`, `format`, `notebooks`
- [x] 0.5 — Create `configs/` YAML files with initial structure
- [x] 0.6 — Verify `pip install -e .` works and `import molgate` succeeds

**Note**: Repo lives at `/home/zhedd/molgate`. Package renamed from `molprop` → `molgate` to match the existing GitHub repository.

**Note**: PyG optional CUDA extensions (`torch-sparse`, `torch-spline-conv`, `torch-scatter`, `torch-cluster`) emit warnings due to ABI mismatch. All suppressed via `warnings.filterwarnings` in `__init__.py`. If new PyG extensions trigger similar warnings, add them to the same filter block.

**Implementation note**: We build and verify each file together. You'll understand every dependency choice and why the package structure looks the way it does.

---

## Phase 1: Data layer

**Goal**: Load any TDC ADMET dataset, compute molecular features, and generate dataset variants.

- [x] 1.1 — `data/registry.py`: A dictionary mapping dataset names to metadata (task type, metric, target column, description). Covers all 22 TDC ADMET datasets. Pure data, no logic. Uses frozen dataclass `DatasetInfo` + `get_dataset_info()` / `list_datasets()` helpers.
- [x] 1.2 — `data/loaders.py`: Functions to download from TDC, canonicalize SMILES, remove duplicates, return clean DataFrames. One function: `load_dataset(name) → DataFrame` (columns: smiles, drug_id, y). RDKit C++ warnings suppressed globally via `RDLogger.DisableLog`.
- [x] 1.3 — `data/featurizer.py`: Three featurization modes:
  - `compute_descriptors(smiles_list) → DataFrame` — 12 curated RDKit 2D descriptors (MW, LogP, TPSA, HBA, HBD, etc.)
  - `compute_fingerprints(smiles_list, radius, nbits) → np.ndarray` — Morgan fingerprints (ECFP4 default)
  - `smiles_to_graph(smiles) → PyG Data` — Atom features (14-dim) + bond features (6-dim), bidirectional edges
- [x] 1.4 — `data/splits.py`: Splitting strategies:
  - `random_split(df, val_frac, test_frac, seed)`
  - `scaffold_split(df, val_frac, test_frac, seed)` — Bemis-Murcko scaffold grouping, train-first greedy assignment
  - `stratified_split(df, val_frac, test_frac, n_bins, seed)` — Quantile-binned proportional splitting
- [x] 1.5 — `data/bias.py`: **Core novelty**. Functions that take a clean dataset and return a biased variant:
  - `bias_by_scaffold(df, top_n)` — Keep only molecules from the N most common scaffolds (acyclics excluded)
  - `bias_by_property_range(df, property_col, low, high)` — Keep only molecules in a property window
  - `bias_by_target_region(df, target_col, quantile_low, quantile_high)` — Drop extremes of target distribution
  - `bias_by_substructure(df, smarts, keep=True)` — Keep/remove molecules matching a SMARTS pattern
  - `bias_by_cluster(df, n_keep, cutoff)` — Butina clustering on Tanimoto distances, keep top-N clusters (replaced single-reference similarity)
  - Each function returns `(biased_df, bias_metadata_dict)` where metadata describes what was removed and why
- [x] 1.6 — Tests for all data layer modules (82 tests passing, `@pytest.mark.slow` for network tests)
- [x] 1.7 — Walked through each function with `if __name__` blocks + IPython embed for interactive inspection

---

## Phase 2: Analysis utilities

**Goal**: Reusable analysis functions that both notebooks and Prefect flows call.

- [x] 2.1 — `analysis/descriptors.py`: Summary statistics (mean, std, skewness, kurtosis), KS test split comparison, Pearson/Spearman descriptor-target correlations, convenience `descriptor_analysis` report
- [x] 2.2 — `analysis/scaffolds.py`: Scaffold extraction (Bemis-Murcko, None for acyclics), diversity metrics (scaffold ratio, singleton fraction, top-K coverage), Jaccard overlap between splits, per-scaffold target stats, frequency table
- [x] 2.3 — `analysis/cliffs.py`: Activity cliff detection via BulkTanimotoSimilarity + activity threshold (auto or manual), CliffPair dataclass, cliff summary metrics, cliff_analysis report. Led to discovery of salt artifacts → added desalting to loaders.
- [x] 2.4 — `analysis/bias_diagnostics.py`: Four diagnostic dimensions:
  - Descriptor distribution shift (KS per descriptor)
  - Scaffold diversity change (ratio, singletons, top-10 coverage)
  - Target distribution shift (mean, std, skewness, range, KS)
  - Adversarial validation (RF on Morgan FPs, 5-fold CV AUROC)
  - Combined `bias_report` for W&B logging
- [x] 2.5 — Tests for analysis modules (145 total tests passing across all modules)

---

## Phase 3: Model layer

**Goal**: Trainable models with a unified interface.

- [x] 3.1 — `models/baseline.py`: `FingerprintModel` class wrapping LightGBM (and optionally XGBoost). Interface: `fit(X, y)`, `predict(X)`, `get_params()`, `feature_importances()`, `tune(X, y, n_trials)`. Handles both classification and regression. Optuna TPE tuning with 9-param search space. Tested: LightGBM-descriptors RMSE=1.10, LightGBM-fingerprints RMSE=1.30 on solubility (descriptors win for solubility due to global physicochemical property correlation).
- [x] 3.2 — `models/gnn.py`: `MoleculeGNN` (PyG GINEConv). Architecture: AtomEncoder(14→128) + BondEncoder(6→128) → 3× [GINEConv(2-layer internal MLP) + BatchNorm + ReLU + Dropout + Residual] → GlobalMeanPool → MLP head(128→128→1). ~120K params. GINEConv chosen over GCN for edge feature support and WL-1 expressiveness. Visualization script at `scripts/visualize_gnn_graph.py`.
- [x] 3.3 — `models/factory.py`: `create_model(model_name, task_type)` reads from `configs/models.yaml`. Supports `overrides` dict for HP sweeps via `_deep_merge`. Attaches featurization metadata (`feature_type`, `fp_radius`, `fp_nbits`) to FingerprintModel and `training_config` to GNN for downstream use by trainer/flows.
- [x] 3.4 — `training/metrics.py`: `compute_metrics(y_true, y_pred, task_type) → dict`. Regression: RMSE, MAE, R², Pearson r. Classification: AUROC, accuracy, F1, precision, recall. All handle edge cases (constant predictions → NaN, single-class → NaN for AUROC). Classification expects probabilities, thresholds at 0.5 for hard metrics.
- [x] 3.5 — `training/trainer.py`: `Trainer` class for GNN. AdamW optimizer, ReduceLROnPlateau scheduler, early stopping with patience. `Trainer.fit(train_graphs, val_graphs) → TrainHistory`. Restores best model state after training. `Trainer.from_config(model, training_config)` for factory integration. MSELoss for regression, BCELoss for classification.
- [x] 3.6 — `training/evaluate.py`: `evaluate_model(model, test_data, y_true, task_type, smiles) → (metrics_dict, predictions_df)`. Returns both aggregate metrics and per-molecule predictions for error analysis. Also includes `evaluate_tdc` and `evaluate_tdc_multi_seed` helpers.
- [x] 3.7 — Tests for model and training modules (test_models.py: 32 tests, test_training.py: 24 tests, all passing)

---

## Phase 4: W&B integration

**Goal**: Every experiment is tracked, comparable, and reproducible.

- [x] 4.1 — `configs/wandb.yaml`: Project name, entity, default tags
- [x] 4.2 — W&B logging strategy implemented in `tracking.py` (not a separate module -- integrated into trainer and flows):
  - **Run config**: dataset name, bias type, bias params, model type, hyperparameters, split type, seed
  - **Run metrics**: train/val loss per epoch, final test metrics
  - **Run artifacts**: per-molecule predictions CSV, best model checkpoint
  - **Run tags**: dataset name, model type, bias variant (e.g., `["solubility", "lgbm", "bias:scaffold_top5"]`)
  - **W&B Tables**: prediction tables with SMILES for interactive exploration
- [x] 4.3 — Test W&B integration with a dry-run (offline mode) — test_wandb.py: 17 tests, all passing

---

## Phase 5: Prefect flows

**Goal**: Orchestrate the full experiment matrix. This is where the tools come together.

- [x] 5.1 — `flows/eda_flow.py`: Full EDA pipeline — load, split, descriptor stats, target stats, scaffold analysis, cliff detection, adversarial validation, W&B logging.
- [x] 5.2 — `flows/train_flow.py`: Single training run — load, bias, diagnostics, split, featurize, train, evaluate, W&B log. Supports all 3 model types (lgbm_morgan, lgbm_descriptors, gnn).
- [x] 5.3 — `flows/bias_experiment.py`: Full experiment matrix. Config-driven from `bias_experiments.yaml`. Sequential execution over N datasets × M conditions × K models × S seeds. Returns aggregated results DataFrame.
- [x] 5.4 — `flows/compare_flow.py`: Accepts results DataFrame directly (or queries W&B). Builds pivot table, degradation columns, heatmap, degradation bar chart. Saves CSV + plots, logs to W&B.
- [x] 5.5 — `tests/test_flows.py`: 41 tests covering all 4 flows — task-level unit tests + end-to-end flow tests with mocked `load_dataset`. GNN flow test marked `@pytest.mark.slow`.

**Note**: Before running the real bias experiments, need to decide: W&B mode (online vs offline), which datasets, number of seeds, and whether to run GNN in the matrix or lgbm-only first.

---

## Phase 6: Ensemble exploration

**Goal**: Systematically test model ensembles to identify the best predictive setup. Inspired by the Titanic stacking lab (`kaggle/titanic/stacking_lab.ipynb`).

- [x] 6.1 — `models/ensemble.py`: Ensemble classes with unified interface:
  - `VotingEnsemble` — Average predictions from multiple base models (simplest)
  - `BlendingEnsemble` — Train base models on train set, learn blend weights on validation set
  - `StackingEnsemble` — K-fold out-of-fold predictions from base models feed a meta-learner (Ridge/LightGBM)
  - `GNNModelWrapper` — bridges MoleculeGNN into the unified fit/predict interface; uses internal 10% val split for early stopping; `requires_graphs=True` triggers hybrid stacking path
  - All support both regression and classification task types
- [x] 6.2 — Extend `models/factory.py` to support ensemble configs; added `base_models` kwarg to `create_model()` for runtime base model injection (bypasses YAML `base_models` list)
- [x] 6.3 — `06_ensemble_lab.ipynb`: Full ensemble exploration:
  - Base model tuning (Optuna, 100 trials for LightGBM, 50 for RF)
  - GNN standalone baseline (GINEConv, MAE=0.819)
  - Prediction correlation heatmap
  - Voting / blending / stacking comparison (untuned + tuned)
  - Two-stage Optuna ensemble tuning via `tune_and_compare()`
  - TDC-compliant 5-seed evaluation (Section 11)
  - Model serialization + round-trip verification (Section 12)
- [x] 6.4 — Best configuration identified: **tuned BlendingEnsemble** (lgbm_morgan + lgbm_descriptors + rf_descriptors + GNN), **MAE = 0.7918 ± 0.0087** (5 TDC seeds). Checkpoint saved to `checkpoints/solubility/blending_tuned/`.
- [ ] 6.5 — Tests for ensemble module (deferred; covered by notebook integration tests)

---

## Phase 7: Predictive API and CLI

**Goal**: Ship the best model(s) as a usable prediction tool — both programmatic (API) and command-line.

- [x] 7.1 — `serve/checkpoint.py`: Save/load trained models. Joblib for tree models, torch `.pt` for GNN (state dict + config), combined directory layout for ensembles (`manifest.json`, `ensemble_meta.json`, `meta.joblib`, `base/`). Round-trip verified bit-exact.
- [x] 7.2 — `serve/predict.py`: Core prediction logic — load checkpoint, dispatch features per model type (reads `feature_map` from `ensemble_meta.json`, computes each feature type once), return `pd.DataFrame(smiles, prediction, valid)`. Invalid SMILES return `NaN`.
- [ ] 7.3 — `serve/app.py`: FastAPI application (**deferred post-PXR challenge**):
  - `POST /predict` — accepts SMILES string or list, returns predictions
  - `GET /health` — health check
  - `GET /models` — list available trained models
- [x] 7.4 — `cli.py`: CLI entrypoint registered via `[project.scripts]`:
  - `molgate predict <property> "CCO"` — single prediction; property = dataset name or `all`
  - `molgate predict <property> --file molecules.csv [--output preds.csv]` — batch
  - `molgate predict all "CCO"` — predict all properties with checkpoints, prints `property: value` per line
  - `molgate models list` — ranked table of all checkpoints with MAE
  - Checkpoint root auto-resolved from package location; overridable via `--checkpoint-root`
- [ ] 7.5 — **Distribution architecture** — makes `molgate` pip-installable with downloadable models:

  **Checkpoint directory resolution hierarchy** (checked in order):
  1. `MOLGATE_CHECKPOINT_DIR` env var — power users, CI, Docker
  2. `~/.molgate/checkpoints/` — standard user installation
  3. `<repo>/checkpoints/` — dev fallback (editable install only)

  **Model hosting**: HuggingFace Hub repo `asinansaglam/molgate-models`. Layout mirrors local checkpoint structure (`solubility/blending_tuned/`, `pxr/blending_tuned/`, ...). `huggingface_hub` added as package dependency.

  **`molgate init` command**:
  - `molgate init` — interactive: show available vs installed, prompt to download
  - `molgate init --all` — download all available models
  - `molgate init solubility pxr` — download specific properties
  - `molgate init --list` — compare remote (HF Hub) vs local (`~/.molgate/`) without downloading
  - `molgate init --force solubility` — re-download even if already present
  - Idempotent by default; prints download progress

- [ ] 7.6 — Tests for serve and CLI modules (deferred post-PXR)

---

## Phase 8: Notebooks

**Goal**: GitHub-rendered notebooks that tell the story. These pull from W&B and the analysis modules -- they don't run training themselves.

- [ ] 8.1 — `01_data_overview.ipynb`: Landscape of ADMET datasets. Dataset sizes, task types, target distributions. Chemical space visualization. Scaffold diversity comparison. This is the revised version of `molecule_property/data_analysis.ipynb`, cleaned up and with narrative.
- [ ] 8.2 — `02_solubility_deep_dive.ipynb`: Deep EDA for solubility. Target analysis, descriptor profiling, scaffold analysis, activity cliffs, adversarial validation, fingerprint baselines. Revised from `molecule_property/solubility_analysis.ipynb`.
- [ ] 8.3 — `03_bias_design.ipynb`: **New notebook**. Explains and visualizes each bias strategy. Shows what each biased dataset looks like vs the original (distribution overlays, scaffold diversity bars, chemical space t-SNE colored by included/excluded). This is the "methods" section of the story.
- [ ] 8.4 — `04_model_comparison.ipynb`: Pulls results from W&B. Model performance across datasets. Compares fingerprint baselines vs GNN. Discusses when GNN is worth the complexity.
- [ ] 8.5 — `05_bias_impact.ipynb`: **The key results notebook**. Pulls bias experiment results from W&B. Heatmaps: bias condition × model type → metric. Answers: Which biases hurt most? Does GNN degrade differently than LightGBM under bias? Is scaffold diversity or target range more important?
- [ ] 8.6 — `06_ensemble_lab.ipynb`: Ensemble exploration results — which combinations work best, prediction correlation heatmaps, stacking performance curves.
- [ ] 8.7 — `07_conclusions.ipynb`: Summary of findings, practical recommendations for dataset curation in molecular ML, limitations, future directions.
- [ ] 8.8 — Ensure all notebooks render cleanly on GitHub (no widget outputs, static plots, markdown narrative between cells)

---

## Phase 9: Polish and CI

- [ ] 9.1 — `README.md`: Project overview, key results figure (the bias impact heatmap), installation instructions including `molgate init`, how to reproduce, link to notebooks
- [ ] 9.2 — `.github/workflows/ci.yml`: Lint (ruff) + type check (mypy) + tests (pytest, no slow/gpu)
- [ ] 9.3 — Verify full reproducibility: clone → install → `molgate init` → `molgate predict solubility "CCO"` works; separately: `prefect run bias_experiment` → notebooks pull results → all renders
- [ ] 9.4 — Docker setup for the prediction API (Dockerfile + docker-compose.yaml)
- [ ] 9.5 — PyPI packaging: verify `pip install molgate` from PyPI works end-to-end; add `huggingface_hub` to `[project.dependencies]`; confirm `molgate init` + `molgate predict` work in a clean venv with no local checkout

---

## Phase 10: OpenADMET PXR Blind Challenge

**Goal**: Adapt the molgate pipeline to compete in the [OpenADMET PXR Blind Challenge](https://huggingface.co/spaces/openadmet/pxr-challenge).  Predict **pEC50** (potency, −log₁₀ molarity) for PXR (Pregnane X Receptor) activation on 513 blind test compounds, trained on 4,140 labelled molecules.  A secondary stretch goal is joint prediction of **Emax** (maximum induction effect).

**Why this is a natural next step**: Every component built in Phases 1–9 transfers directly — featurization, models, ensembles, tuning, bias analysis, serving.  The only new code needed is a HuggingFace data loader and a submission script.  The PXR dataset is structurally identical to TDC datasets (SMILES → continuous target, regression).

**Dataset**: `openadmet/pxr-challenge-train-test` on HuggingFace  
- Train: 4,140 molecules | Test: 513 molecules (blind)  
- Primary target: `pEC50` (float, range 1.61–7.55)  
- Secondary targets: `Emax_estimate`, `Emax.vs.pos.ctrl_estimate` (with CI columns)  
- Split column: `Split` ("train"/"test")

**Evaluation**: Standard regression metrics (RMSE, MAE, R²) on blind test set, submitted via the HuggingFace Space leaderboard as a CSV.

### Tasks

- [ ] 10.1 — **HuggingFace data loader**: Add `load_hf_dataset(repo_id, smiles_col, target_col, split_col)` to `data/loaders.py`.  Applies the same desalting + canonicalization + deduplication pipeline as `load_dataset()`.  Returns `(train_df, test_df)` with standard `smiles / y` columns.  Add `"pxr"` entry to the dataset registry (or handle as an ad-hoc dataset without registry).

- [ ] 10.2 — **EDA on PXR training set**: Run `eda_flow.py` on the PXR training data.  Key questions: target distribution shape (is pEC50 bimodal?), scaffold diversity vs. TDC solubility, activity cliff density, adversarial AUROC of train vs. test (are they from the same chemical space?).  Log to W&B under a new `pxr` project.

- [ ] 10.3 — **Bias characterisation**: Apply the existing bias experiment matrix to the PXR training set — scaffold, MW range, target region, cluster biases — to understand which training data compositions hurt generalisation.  This is the scientific angle that differentiates the submission from a vanilla ML entry.

- [ ] 10.4 — **Base model sweep**: Train all registry models (lgbm_morgan, lgbm_descriptors, rf_morgan, ridge_descriptors, svr_descriptors, gnn) on the PXR training set using the TDC-style fixed split (use the provided `Split` column directly, no random re-splitting).  Record pEC50 MAE on the blind test set via the leaderboard.

- [ ] 10.5 — **Ensemble lab (pEC50)**: Run the ensemble lab notebook (adapted from `06_ensemble_lab.ipynb`) on PXR.  Tune voting / blending / stacking ensembles with `tune_and_compare()`.  Identify the best single-target configuration.

- [ ] 10.6 — **Multi-output stretch goal**: Add `MultiOutputSklearnModel` or use `sklearn.multioutput.MultiOutputRegressor` to jointly predict `pEC50 + Emax_estimate`.  Evaluate whether joint training improves pEC50 MAE (shared representation may help for correlated targets).

- [ ] 10.7 — **Submission script**: Write `scripts/submit_pxr.py` — loads the best serialised model, predicts on the blind test SMILES, formats output as `{Molecule Name, pEC50_pred}` CSV, and validates column names match leaderboard expectations.  Wrap with `argparse` (`--model`, `--output`).  Final user-facing shape: `molgate init pxr && molgate predict pxr "CCO"` — checkpoint lands in `~/.molgate/checkpoints/pxr/` and CLI picks it up automatically.

- [ ] 10.8 — **Leaderboard submission and iteration**: Submit initial predictions.  Review leaderboard position.  Iterate on: (a) base model tuning with more Optuna trials, (b) adding more ensemble members, (c) data augmentation (stereoisomers, tautomers via RDKit), (d) GNN architecture search.

- [ ] 10.9 — **Notebook write-up** (`notebooks/08_pxr_challenge.ipynb`): Narrative notebook documenting the approach — EDA, bias analysis, model selection, final submission.  Structured as a mini-paper: motivation → methods → results → discussion.  Publishable as a HuggingFace model card or blog post.

### Key reuse from existing work

| Existing component | PXR use |
|---|---|
| `data/featurizer.py` | Zero changes — SMILES → FP/desc/graphs |
| `models/` (all) | Zero changes — same regression interface |
| `models/ensemble.py` + `tuning.py` | Zero changes — same API |
| `flows/train_flow.py` | Minor: accept pre-split DataFrames (already supported via `preloaded_df` + `preloaded_test_df`) |
| `flows/bias_experiment.py` | Zero changes — config-driven |
| `flows/compare_flow.py` | Zero changes |
| `serve/` (Phase 7) | Zero changes — serialise winner, serve via existing API |
| W&B tracking | New project `pxr-challenge`, same tags/tables pattern |

### New code required

| File | Change |
|---|---|
| `data/loaders.py` | Add `load_hf_dataset()` |
| `data/registry.py` | Add `"pxr"` entry (or ad-hoc path) |
| `scripts/submit_pxr.py` | New submission CLI |
| `notebooks/08_pxr_challenge.ipynb` | New challenge notebook |
| `configs/bias_experiments.yaml` | Add `pxr` dataset block |

### CV/interview value added

Competing in an open blind challenge demonstrates that the pipeline is not just a toy — it produces submissions comparable to other teams using the same evaluation.  The bias analysis angle (showing *why* certain training compositions matter for PXR generalisation) is a unique scientific contribution beyond the leaderboard number.

---

## Implementation approach

**Every implementation step follows this pattern:**

1. **Explain** what the file/class/function does and why it exists
2. **Write** the code together, discussing design choices
3. **Test** immediately -- either a unit test or a REPL verification
4. **Commit** the working piece before moving on

**Pacing**: One module per session. No rushing. If a function needs 20 lines, we write and understand all 20 lines. If a design choice has tradeoffs, we discuss them.

**Order of implementation**: Phases 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10, strictly sequential. Within each phase, tasks are sequential. We don't start Phase 2 until Phase 1 is tested and working.

**Scope management**: Start with **solubility only** as the primary dataset through Phases 1-5. Once the full pipeline works end-to-end for solubility, extend to 2-3 more properties (likely LD50 and Ames to cover regression + classification). Don't try to support all 22 datasets upfront.

---

## What this showcases on a CV / in an interview

| Skill | Evidence |
|---|---|
| ML pipeline engineering | End-to-end from raw data to model comparison |
| Experiment tracking | W&B integration with structured tagging and comparison dashboards |
| Workflow orchestration | Prefect flows managing a matrix of experiments |
| Domain expertise | Molecular featurization, scaffold splits, activity cliffs, ADMET knowledge |
| Scientific rigor | Systematic bias experiments, proper validation, reproducibility |
| Communication | Narrative notebooks explaining findings to a non-specialist |
| Software engineering | Clean package structure, tests, CI, typed Python |
| Model selection & ensembling | Systematic ensemble exploration (stacking, blending, voting) |
| Production readiness | Predictive API (FastAPI) + CLI entrypoint for trained models |
| Open challenge participation | OpenADMET PXR blind challenge — end-to-end from EDA to leaderboard submission |
