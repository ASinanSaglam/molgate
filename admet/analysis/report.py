"""Load and aggregate experiment results for analysis and presentation.

Consumes the JSON files written to results/runs/ by runner.py and
produces summary DataFrames (and CSV files) in results/tables/.

Designed to be called from notebooks/02_split_comparison.ipynb and
notebooks/03_bias_analysis.ipynb, or from the run_bias_study.py CLI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TABLES_DIR = Path(__file__).parent.parent.parent / "results" / "tables"


def load_results(
    results_dir: Path,
    property_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Load all run JSONs from results/runs/ into a tidy DataFrame.

    Args:
        results_dir: Directory containing run JSON files (results/runs/).
        property_filter: If given, only load results for these property names.

    Returns:
        Tidy DataFrame with one row per run. Columns include: property,
        split_method, bias_type, seed, train_size, test_metric values, etc.
    """
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        logger.warning("No result JSON files found in %s", results_dir)
        return pd.DataFrame()

    rows = []
    for path in json_files:
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON %s: %s", path.name, e)
                continue

        if property_filter and data.get("property") not in property_filter:
            continue
        rows.append(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Flatten bias_params into separate columns for easier filtering
    if "bias_params" in df.columns:
        bias_expanded = pd.json_normalize(df["bias_params"].fillna({}).tolist())
        bias_expanded.columns = [f"bias_{c}" for c in bias_expanded.columns]
        df = pd.concat([df.drop(columns=["bias_params"]), bias_expanded], axis=1)

    return df


def make_split_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to: property × split_method → mean ± std of primary test metric.

    Aggregates over seeds. Only baseline runs (bias_type == 'no_bias') are
    included, so this table isolates the effect of split strategy alone.

    Returns:
        DataFrame with MultiIndex columns (split_method, stat) where stat
        is 'mean' or 'std'. Index is property name.
    """
    baseline = df[df["bias_type"] == "no_bias"].copy()
    if baseline.empty:
        logger.warning("No baseline (no_bias) runs found.")
        return pd.DataFrame()

    metric_col = _infer_primary_metric_col(baseline)
    pivot = (
        baseline
        .groupby(["property", "split_method"])[metric_col]
        .agg(["mean", "std"])
        .unstack("split_method")
    )
    return pivot


def make_bias_sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to: property × bias_type → delta from no_bias baseline.

    Delta is (biased_metric - baseline_metric), averaged over seeds and splits.
    Positive delta = improvement over baseline (for AUROC/R²);
    negative delta = degradation. Sign convention is consistent with higher=better.

    Returns:
        DataFrame with property as index and bias_type as columns.
    """
    metric_col = _infer_primary_metric_col(df)

    # Compute per-(property, split, seed) baseline
    baseline = df[df["bias_type"] == "no_bias"].copy()
    baseline_mean = (
        baseline
        .groupby(["property", "split_method"])[metric_col]
        .mean()
        .rename("baseline")
    )

    biased = df[df["bias_type"] != "no_bias"].copy()
    if biased.empty:
        logger.warning("No biased runs found in results.")
        return pd.DataFrame()

    merged = biased.merge(
        baseline_mean.reset_index(),
        on=["property", "split_method"],
        how="left",
    )
    merged["delta"] = merged[metric_col] - merged["baseline"]

    pivot = (
        merged
        .groupby(["property", "bias_type"])["delta"]
        .mean()
        .unstack("bias_type")
    )
    return pivot


def save_tables(results_dir: Path, tables_dir: Path | None = None) -> None:
    """Load results and write both summary tables to CSV.

    Args:
        results_dir: Path to results/runs/.
        tables_dir: Output directory (defaults to results/tables/).
    """
    if tables_dir is None:
        tables_dir = _TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        logger.warning("No results to summarize.")
        return

    split_table = make_split_comparison_table(df)
    bias_table = make_bias_sensitivity_table(df)

    split_path = tables_dir / "split_comparison.csv"
    bias_path = tables_dir / "bias_sensitivity.csv"

    split_table.to_csv(split_path)
    bias_table.to_csv(bias_path)

    logger.info("Saved split comparison table to %s", split_path)
    logger.info("Saved bias sensitivity table to %s", bias_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_primary_metric_col(df: pd.DataFrame) -> str:
    """Infer which test metric column to use as the primary metric.

    Priority: test_auroc > test_rmse > test_r2 > test_spearman > test_mae.
    Falls back to the first column starting with 'test_'.
    """
    candidates = ["test_auroc", "test_rmse", "test_r2", "test_spearman", "test_mae"]
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    test_cols = [c for c in df.columns if c.startswith("test_")]
    if test_cols:
        return test_cols[0]
    raise ValueError("No test metric columns found in results DataFrame.")
