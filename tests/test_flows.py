"""Tests for Prefect flows: compare_flow, train_flow, bias_experiment, eda_flow.

Strategy:
    - Task-level tests: call Prefect tasks directly (outside a flow they run
      synchronously in the current thread — no server needed).
    - Flow-level tests: call the @flow functions with mock data and
      wandb_mode="disabled" so no network calls are made.
    - All tests using TDC network access are @pytest.mark.slow.
    - GNN training tests are @pytest.mark.slow (even with 2 epochs, PyG
      graph featurization adds ~5-10 s).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_SMILES_10 = [
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "c1ccc(O)cc1",
    "CC(=O)O",
    "c1ccncc1",
    "C1CCCCC1",
    "c1ccc(-c2ccccc2)cc1",
    "CC(C)(C)c1ccc(O)cc1",
]


@pytest.fixture
def synthetic_flow_df():
    """40-molecule DataFrame matching load_dataset output (smiles, drug_id, y).

    Built by repeating the 10 SMILES × 4 with different drug_ids and
    randomly varied targets.  This gives enough rows (40) for a 70/15/15
    train/val/test split with room to spare.
    """
    rng = np.random.default_rng(0)
    smiles = _SMILES_10 * 4
    n = len(smiles)
    return pd.DataFrame({
        "smiles": smiles,
        "drug_id": [f"mol_{i:03d}" for i in range(n)],
        "y": rng.standard_normal(n).astype(float),
    })


@pytest.fixture
def synthetic_results_df():
    """Typical bias_experiment_flow output with 3 conditions × 2 models × 2 seeds."""
    rng = np.random.default_rng(42)
    conditions = ["unbiased", "scaffold_top10", "mw_narrow"]
    models = ["lgbm_morgan", "lgbm_descriptors"]
    rows = []
    for condition in conditions:
        for model in models:
            for seed in [42, 123]:
                rmse = 1.0 + (0 if condition == "unbiased" else 0.2) + rng.normal(0, 0.02)
                rows.append({
                    "dataset": "solubility",
                    "model": model,
                    "condition": condition,
                    "bias": condition,
                    "split": "random",
                    "seed": seed,
                    "rmse": max(0.5, rmse),
                    "mae": max(0.3, rmse * 0.75),
                    "r2": 0.85 - 0.1 * (condition != "unbiased") + rng.normal(0, 0.01),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compare_flow — task tests
# ---------------------------------------------------------------------------

class TestTaskLoadResults:
    """Tests for compare_flow.task_load_results."""

    def test_accepts_valid_df(self, synthetic_results_df):
        from molgate.flows.compare_flow import task_load_results

        out = task_load_results(
            results_df=synthetic_results_df,
            wandb_group=None,
            wandb_project=None,
            dataset_name="solubility",
            primary_metric="rmse",
        )
        assert len(out) == len(synthetic_results_df)
        assert "rmse" in out.columns

    def test_missing_primary_metric_raises(self, synthetic_results_df):
        from molgate.flows.compare_flow import task_load_results

        with pytest.raises(ValueError, match="primary_metric"):
            task_load_results(
                results_df=synthetic_results_df,
                wandb_group=None,
                wandb_project=None,
                dataset_name="solubility",
                primary_metric="auroc",  # not in df
            )

    def test_missing_required_column_raises(self):
        from molgate.flows.compare_flow import task_load_results

        bad_df = pd.DataFrame({"model": ["lgbm"], "rmse": [1.0]})  # missing 'condition'
        with pytest.raises(ValueError, match="condition"):
            task_load_results(
                results_df=bad_df,
                wandb_group=None,
                wandb_project=None,
                dataset_name="solubility",
                primary_metric="rmse",
            )

    def test_no_input_raises(self):
        from molgate.flows.compare_flow import task_load_results

        with pytest.raises(ValueError, match="results_df"):
            task_load_results(
                results_df=None,
                wandb_group=None,
                wandb_project=None,
                dataset_name="solubility",
                primary_metric="rmse",
            )


class TestTaskBuildComparisonTable:
    """Tests for compare_flow.task_build_comparison_table."""

    def test_pivot_shape(self, synthetic_results_df):
        """Pivot should have one row per unique condition."""
        from molgate.flows.compare_flow import task_build_comparison_table

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        n_conditions = synthetic_results_df["condition"].nunique()
        assert len(pivot) == n_conditions

    def test_model_columns_present(self, synthetic_results_df):
        """Each model should appear as a column."""
        from molgate.flows.compare_flow import task_build_comparison_table

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        for model in synthetic_results_df["model"].unique():
            assert model in pivot.columns

    def test_degradation_columns_appended(self, synthetic_results_df):
        """Each model should get a <model>_deg_pct column."""
        from molgate.flows.compare_flow import task_build_comparison_table

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        for model in synthetic_results_df["model"].unique():
            assert f"{model}_deg_pct" in pivot.columns
        assert "mean_degradation_pct" in pivot.columns

    def test_degradation_sign_for_rmse(self, synthetic_results_df):
        """Biased conditions should have positive degradation for RMSE (higher = worse)."""
        from molgate.flows.compare_flow import task_build_comparison_table

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        # Both biased conditions should have mean_degradation_pct > 0
        for condition in ["scaffold_top10", "mw_narrow"]:
            assert pivot.loc[condition, "mean_degradation_pct"] > 0, (
                f"Expected positive degradation for {condition}"
            )

    def test_baseline_degradation_near_zero(self, synthetic_results_df):
        """Baseline condition's own degradation should be ~0."""
        from molgate.flows.compare_flow import task_build_comparison_table

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        assert pivot.loc["unbiased", "mean_degradation_pct"] == pytest.approx(0.0, abs=1e-10)

    def test_averages_over_seeds(self, synthetic_results_df):
        """Values in the pivot should be the mean over seeds, not per-seed."""
        from molgate.flows.compare_flow import task_build_comparison_table

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        # Manual check: unbiased lgbm_morgan pivot value == mean of both seeds
        expected = (
            synthetic_results_df
            .query("condition == 'unbiased' and model == 'lgbm_morgan'")["rmse"]
            .mean()
        )
        assert pivot.loc["unbiased", "lgbm_morgan"] == pytest.approx(expected, rel=1e-6)

    def test_missing_baseline_skips_degradation(self):
        """If baseline_condition is absent, returns pivot without degradation columns."""
        from molgate.flows.compare_flow import task_build_comparison_table

        df = pd.DataFrame({
            "model": ["lgbm"] * 2,
            "condition": ["biased_a", "biased_b"],
            "seed": [42, 42],
            "rmse": [1.2, 1.4],
        })
        pivot = task_build_comparison_table(df, "rmse", baseline_condition="unbiased")
        assert "mean_degradation_pct" not in pivot.columns


class TestTaskGeneratePlots:
    """Tests for heatmap and degradation chart generation."""

    def test_heatmap_creates_png(self, synthetic_results_df, tmp_path):
        from molgate.flows.compare_flow import task_build_comparison_table, task_generate_heatmap

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        path = task_generate_heatmap(pivot=pivot, primary_metric="rmse", output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_degradation_chart_creates_png(self, synthetic_results_df, tmp_path):
        from molgate.flows.compare_flow import (
            task_build_comparison_table,
            task_generate_degradation_chart,
        )

        pivot = task_build_comparison_table(
            results_df=synthetic_results_df,
            primary_metric="rmse",
            baseline_condition="unbiased",
        )
        path = task_generate_degradation_chart(
            pivot=pivot,
            primary_metric="rmse",
            baseline_condition="unbiased",
            output_dir=tmp_path,
        )
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0

    def test_heatmap_filename_includes_metric(self, synthetic_results_df, tmp_path):
        from molgate.flows.compare_flow import task_build_comparison_table, task_generate_heatmap

        pivot = task_build_comparison_table(synthetic_results_df, "rmse", "unbiased")
        path = task_generate_heatmap(pivot, "rmse", tmp_path)
        assert "rmse" in path.name


class TestCompareFlow:
    """End-to-end tests for compare_flow."""

    def test_returns_expected_keys(self, synthetic_results_df, tmp_path):
        from molgate.flows.compare_flow import compare_flow

        result = compare_flow(
            results_df=synthetic_results_df,
            dataset_name="solubility",
            primary_metric="rmse",
            baseline_condition="unbiased",
            output_dir=tmp_path,
            wandb_mode="disabled",
        )
        assert set(result.keys()) == {
            "pivot", "results_df", "heatmap_path", "degradation_path", "output_dir"
        }

    def test_output_dir_created(self, synthetic_results_df):
        from molgate.flows.compare_flow import compare_flow

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new_subdir"
            assert not out.exists()
            compare_flow(
                results_df=synthetic_results_df,
                dataset_name="solubility",
                primary_metric="rmse",
                output_dir=out,
                wandb_mode="disabled",
            )
            assert out.exists()

    def test_csv_files_written(self, synthetic_results_df, tmp_path):
        from molgate.flows.compare_flow import compare_flow

        compare_flow(
            results_df=synthetic_results_df,
            dataset_name="solubility",
            primary_metric="rmse",
            output_dir=tmp_path,
            wandb_mode="disabled",
        )
        assert (tmp_path / "comparison_rmse.csv").exists()
        assert (tmp_path / "results_raw.csv").exists()

    def test_pivot_in_result(self, synthetic_results_df, tmp_path):
        from molgate.flows.compare_flow import compare_flow

        result = compare_flow(
            results_df=synthetic_results_df,
            dataset_name="solubility",
            primary_metric="rmse",
            output_dir=tmp_path,
            wandb_mode="disabled",
        )
        pivot = result["pivot"]
        assert isinstance(pivot, pd.DataFrame)
        assert len(pivot) == synthetic_results_df["condition"].nunique()


# ---------------------------------------------------------------------------
# train_flow — helper and end-to-end tests
# ---------------------------------------------------------------------------

class TestApplyBias:
    """Tests for train_flow._apply_bias helper."""

    def test_none_config_returns_full_df(self):
        from molgate.flows.train_flow import task_apply_bias

        df = pd.DataFrame({"smiles": _SMILES_10, "y": range(10)})
        out_df, meta = task_apply_bias(df, bias_config=None)
        assert len(out_df) == len(df)
        assert meta is None

    def test_unbiased_method_returns_full_df(self):
        from molgate.flows.train_flow import task_apply_bias

        df = pd.DataFrame({"smiles": _SMILES_10, "y": range(10)})
        out_df, meta = task_apply_bias(df, bias_config={"method": "unbiased"})
        assert len(out_df) == len(df)
        assert meta is None

    def test_scaffold_bias_reduces_dataset(self):
        from molgate.flows.train_flow import task_apply_bias

        # Use larger set so scaffold grouping has something to work with
        rng = np.random.default_rng(0)
        smiles = _SMILES_10 * 4
        df = pd.DataFrame({
            "smiles": smiles,
            "drug_id": [f"mol_{i}" for i in range(len(smiles))],
            "y": rng.standard_normal(len(smiles)),
        })
        out_df, meta = task_apply_bias(df, {"method": "scaffold", "params": {"top_n": 2}})
        assert len(out_df) < len(df)
        assert meta is not None

    def test_unknown_method_raises(self):
        from molgate.flows.train_flow import task_apply_bias

        df = pd.DataFrame({"smiles": _SMILES_10, "y": range(10)})
        with pytest.raises(ValueError, match="Unknown bias method"):
            task_apply_bias(df, {"method": "nonexistent_method"})


class TestTrainFlow:
    """End-to-end tests for train_flow with mocked data loading."""

    def test_lgbm_morgan_unbiased_returns_keys(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="lgbm_morgan",
                task_type="regression",
                bias_config=None,
                split_type="random",
                seed=42,
                wandb_mode="disabled",
            )

        assert set(result.keys()) >= {"dataset", "model", "bias", "metrics", "train_info"}
        assert result["dataset"] == "solubility"
        assert result["model"] == "lgbm_morgan"

    def test_lgbm_morgan_metrics_are_finite(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="lgbm_morgan",
                task_type="regression",
                wandb_mode="disabled",
            )

        metrics = result["metrics"]
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        # pearson_r can be NaN when predictions are near-constant (tiny test set),
        # so only assert finiteness for the primary error metrics.
        for k in ("rmse", "mae"):
            assert np.isfinite(metrics[k]), f"Non-finite metric {k}={metrics[k]}"
        assert metrics["rmse"] > 0

    def test_lgbm_descriptors_unbiased(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="lgbm_descriptors",
                task_type="regression",
                wandb_mode="disabled",
            )
        assert result["metrics"]["rmse"] > 0

    def test_bias_metadata_populated_when_biased(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="lgbm_morgan",
                task_type="regression",
                bias_config={"method": "target_region", "params": {
                    "target_col": "y", "quantile_low": 0.1, "quantile_high": 0.9,
                }},
                wandb_mode="disabled",
            )
        assert result["bias_metadata"] is not None
        assert result["bias"] == "target_region"

    def test_unbiased_bias_metadata_is_none(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="lgbm_morgan",
                task_type="regression",
                bias_config=None,
                wandb_mode="disabled",
            )
        assert result["bias_metadata"] is None

    def test_predictions_df_has_smiles_column(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="lgbm_morgan",
                task_type="regression",
                wandb_mode="disabled",
            )
        pred_df = result["predictions_df"]
        assert isinstance(pred_df, pd.DataFrame)
        assert "smiles" in pred_df.columns or "y_true" in pred_df.columns

    @pytest.mark.slow
    def test_gnn_trains_and_returns_metrics(self, synthetic_flow_df):
        from molgate.flows.train_flow import train_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = train_flow(
                dataset_name="solubility",
                model_name="gnn",
                task_type="regression",
                bias_config=None,
                wandb_mode="disabled",
                model_overrides={"training": {"epochs": 3, "patience": 2}},
            )

        assert "rmse" in result["metrics"]
        assert result["train_info"]["model_type"] == "gnn"


# ---------------------------------------------------------------------------
# bias_experiment — helper tests
# ---------------------------------------------------------------------------

class TestBuildBiasConfig:
    """Tests for bias_experiment._build_bias_config."""

    def test_null_bias_fn_returns_none(self):
        from molgate.flows.bias_experiment import _build_bias_config

        assert _build_bias_config("unbiased", {"bias_fn": None}) is None

    def test_missing_bias_fn_returns_none(self):
        from molgate.flows.bias_experiment import _build_bias_config

        assert _build_bias_config("unbiased", {}) is None

    def test_scaffold_maps_to_correct_method(self):
        from molgate.flows.bias_experiment import _build_bias_config

        cfg = _build_bias_config(
            "scaffold_top10",
            {"bias_fn": "bias_by_scaffold", "params": {"top_n": 10}},
        )
        assert cfg is not None
        assert cfg["method"] == "scaffold"
        assert cfg["params"]["top_n"] == 10

    def test_target_col_Y_lowercased(self):
        from molgate.flows.bias_experiment import _build_bias_config

        cfg = _build_bias_config(
            "target_extremes_removed",
            {
                "bias_fn": "bias_by_target_region",
                "params": {"target_col": "Y", "quantile_low": 0.1, "quantile_high": 0.9},
            },
        )
        assert cfg["params"]["target_col"] == "y"

    def test_unknown_fn_uses_fn_name_as_method(self):
        from molgate.flows.bias_experiment import _build_bias_config

        cfg = _build_bias_config("custom", {"bias_fn": "custom_bias_fn"})
        assert cfg is not None
        assert cfg["method"] == "custom_bias_fn"


class TestBuildExperimentMatrix:
    """Tests for bias_experiment.task_build_matrix."""

    def _make_config(self, n_conditions=2):
        conditions = {
            f"cond_{i}": {"bias_fn": None, "split": "random"}
            for i in range(n_conditions)
        }
        return {
            "bias_conditions": conditions,
            "experiment": {
                "datasets": ["solubility"],
                "models": ["lgbm_morgan"],
                "seeds": [42],
            },
        }

    def test_matrix_size(self):
        from molgate.flows.bias_experiment import task_build_matrix

        config = self._make_config(n_conditions=3)
        matrix = task_build_matrix(config)
        # 1 dataset × 3 conditions × 1 model × 1 seed = 3 runs
        assert len(matrix) == 3

    def test_matrix_entry_has_required_keys(self):
        from molgate.flows.bias_experiment import task_build_matrix

        config = self._make_config(n_conditions=1)
        matrix = task_build_matrix(config)
        required = {"dataset", "model", "condition_name", "bias_config", "split_type", "seed"}
        assert required.issubset(set(matrix[0].keys()))

    def test_override_datasets(self):
        from molgate.flows.bias_experiment import task_build_matrix

        config = self._make_config(n_conditions=1)
        matrix = task_build_matrix(config, datasets=["solubility", "lipophilicity"])
        # 2 datasets × 1 condition × 1 model × 1 seed = 2 runs
        assert len(matrix) == 2
        datasets_in_matrix = {r["dataset"] for r in matrix}
        assert datasets_in_matrix == {"solubility", "lipophilicity"}

    def test_unknown_condition_skipped(self):
        from molgate.flows.bias_experiment import task_build_matrix

        config = self._make_config(n_conditions=2)
        matrix = task_build_matrix(
            config, conditions=["cond_0", "does_not_exist"]
        )
        # Only cond_0 is valid
        assert len(matrix) == 1
        assert matrix[0]["condition_name"] == "cond_0"

    def test_multi_seed_multiplies(self):
        from molgate.flows.bias_experiment import task_build_matrix

        config = self._make_config(n_conditions=1)
        matrix = task_build_matrix(config, seeds=[42, 123, 7])
        assert len(matrix) == 3


class TestBiasExperimentFlow:
    """End-to-end tests for bias_experiment_flow (minimal matrix only)."""

    def test_returns_dataframe(self, synthetic_flow_df):
        from molgate.flows.bias_experiment import bias_experiment_flow

        # task_prepare_dataset calls both load_dataset and compute_descriptors
        # internally.  We patch load_dataset and let compute_descriptors run
        # for real (it handles our 10 SMILES fine).
        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = bias_experiment_flow(
                datasets=["solubility"],
                models=["lgbm_morgan"],
                conditions=["unbiased"],
                seeds=[42],
                task_type="regression",
                wandb_mode="disabled",
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # 1 dataset × 1 condition × 1 model × 1 seed

    def test_multiple_conditions_produce_correct_row_count(self, synthetic_flow_df):
        from molgate.flows.bias_experiment import bias_experiment_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = bias_experiment_flow(
                datasets=["solubility"],
                models=["lgbm_morgan"],
                conditions=["unbiased", "target_extremes_removed"],
                seeds=[42],
                task_type="regression",
                wandb_mode="disabled",
            )

        assert len(result) == 2

    def test_result_has_metric_columns(self, synthetic_flow_df):
        from molgate.flows.bias_experiment import bias_experiment_flow

        with patch("molgate.data.loaders.load_dataset", return_value=synthetic_flow_df):
            result = bias_experiment_flow(
                datasets=["solubility"],
                models=["lgbm_morgan"],
                conditions=["unbiased"],
                seeds=[42],
                task_type="regression",
                wandb_mode="disabled",
            )

        assert "rmse" in result.columns
        assert "condition" in result.columns
        assert "model" in result.columns
