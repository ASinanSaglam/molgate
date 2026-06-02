"""Prefect flow: aggregate bias experiment results and generate comparison artifacts.

This flow is the final step in the experiment pipeline — it takes the
results from ``bias_experiment_flow`` (either passed directly or queried
from W&B) and produces:

    1. A comparison DataFrame (conditions × models, primary metric)
    2. A pivot table with per-condition degradation vs the unbiased baseline
    3. A heatmap: conditions (rows) × models (columns), colored by metric
    4. A degradation bar chart: % change from baseline per condition/model
    5. All artifacts logged to W&B and saved to disk

Design decisions:
    - **Two input modes**: accepts ``results_df`` directly (from
      ``bias_experiment_flow``) OR queries the W&B API by group name.
      The direct-pass mode works with ``wandb_mode="disabled"`` and avoids
      a network round-trip during local development.
    - **Sign-aware degradation**: RMSE/MAE degrade when they increase;
      AUROC/R² degrade when they decrease.  The flow derives this from
      ``primary_metric`` so the degradation sign is always "positive = worse."
    - **Baseline-relative**: degradation is computed as percent change from
      the ``baseline_condition`` (default: "unbiased") for each model
      independently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prefect import flow, task

matplotlib.use("Agg")  # non-interactive backend for headless environments

logger = logging.getLogger(__name__)

# Metrics where lower is better (used to compute degradation sign correctly)
_LOWER_IS_BETTER = {"rmse", "mae", "loss"}


def _lower_is_better(metric: str) -> bool:
    return metric.lower() in _LOWER_IS_BETTER


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load_results")
def task_load_results(
    results_df: pd.DataFrame | None,
    wandb_group: str | None,
    wandb_project: str | None,
    dataset_name: str,
    primary_metric: str,
) -> pd.DataFrame:
    """Return a results DataFrame from either a direct input or W&B query.

    If ``results_df`` is provided, returns it after basic validation.
    Otherwise queries W&B for all runs in the given group/project.

    The returned DataFrame must contain at minimum:
        model, condition, seed, <primary_metric>
    """
    if results_df is not None:
        required = {"model", "condition"}
        missing = required - set(results_df.columns)
        if missing:
            raise ValueError(f"results_df is missing required columns: {missing}")
        if primary_metric not in results_df.columns:
            raise ValueError(
                f"primary_metric={primary_metric!r} not found in results_df columns. "
                f"Available: {list(results_df.columns)}"
            )
        logger.info(f"Using provided results_df: {len(results_df)} rows")
        return results_df

    # W&B query path
    if wandb_group is None and wandb_project is None:
        raise ValueError(
            "Either results_df or (wandb_group / wandb_project) must be provided."
        )

    import wandb

    api = wandb.Api()
    project = wandb_project or "molgate"

    filters: dict[str, Any] = {}
    if wandb_group:
        filters["group"] = wandb_group
    if dataset_name:
        filters["tags"] = {"$in": [dataset_name]}

    logger.info(f"Querying W&B: project={project}, filters={filters}")
    runs = api.runs(project, filters=filters)

    rows = []
    for run in runs:
        row = {
            "model": run.config.get("model", run.name),
            "condition": run.config.get("bias_method", "unbiased"),
            "dataset": run.config.get("dataset", dataset_name),
            "split": run.config.get("split_type", "random"),
            "seed": run.config.get("seed", 0),
        }
        # Pull summary metrics (W&B stores test metrics with "test_" prefix)
        for key, val in run.summary.items():
            metric = key.removeprefix("test_")
            if isinstance(val, (int, float)):
                row[metric] = val
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No W&B runs found for project={project!r}, filters={filters}"
        )

    df = pd.DataFrame(rows)
    logger.info(f"Queried {len(df)} runs from W&B")
    return df


@task(name="build_comparison_table")
def task_build_comparison_table(
    results_df: pd.DataFrame,
    primary_metric: str,
    baseline_condition: str,
) -> pd.DataFrame:
    """Build a pivot table with conditions as rows and models as columns.

    Each cell is the mean of ``primary_metric`` over seeds.  Two extra
    columns are added:
        ``<metric>_vs_baseline``: absolute change from the baseline condition
        ``degradation_pct``: percent degradation (positive = worse performance)

    Returns
    -------
    pd.DataFrame
        Index: condition names.  Columns: one per model + degradation columns.
    """
    # Average over seeds first
    agg = (
        results_df
        .groupby(["condition", "model"])[primary_metric]
        .mean()
        .reset_index()
    )

    pivot = agg.pivot(index="condition", columns="model", values=primary_metric)
    pivot.columns.name = None
    pivot.index.name = "condition"

    model_cols = list(pivot.columns)

    # Compute per-model degradation relative to baseline
    if baseline_condition not in pivot.index:
        logger.warning(
            f"Baseline condition {baseline_condition!r} not in results. "
            f"Available: {list(pivot.index)}.  Skipping degradation columns."
        )
        return pivot

    baseline_row = pivot.loc[baseline_condition, model_cols]

    if _lower_is_better(primary_metric):
        # RMSE: degradation = (biased - baseline) / |baseline| * 100
        # Positive degradation means model got worse (higher error)
        degradation = ((pivot[model_cols] - baseline_row) / baseline_row.abs()) * 100
    else:
        # AUROC / R²: degradation = (baseline - biased) / |baseline| * 100
        degradation = ((baseline_row - pivot[model_cols]) / baseline_row.abs()) * 100

    # Append a mean-degradation column (across models) for sorting
    pivot["mean_degradation_pct"] = degradation.mean(axis=1)

    # Append per-model degradation as separate columns
    for col in model_cols:
        pivot[f"{col}_deg_pct"] = degradation[col]

    logger.info(
        f"Comparison table: {len(pivot)} conditions × {len(model_cols)} models\n"
        + pivot[model_cols].to_string()
    )
    return pivot


@task(name="generate_heatmap")
def task_generate_heatmap(
    pivot: pd.DataFrame,
    primary_metric: str,
    output_dir: Path,
    title: str = "",
) -> Path:
    """Generate a condition × model heatmap of the primary metric.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    model_cols = [c for c in pivot.columns if not c.endswith("_deg_pct") and c != "mean_degradation_pct"]

    metric_data = pivot[model_cols].copy()

    # Sort rows: baseline first, then by mean degradation (worst last)
    if "mean_degradation_pct" in pivot.columns:
        order = pivot["mean_degradation_pct"].sort_values().index
        metric_data = metric_data.loc[order]

    # Choose colormap: lower-is-better → green at bottom, red at top (reversed)
    cmap = "RdYlGn_r" if _lower_is_better(primary_metric) else "RdYlGn"

    fig, ax = plt.subplots(figsize=(max(6, len(model_cols) * 2), max(5, len(metric_data) * 0.7)))

    sns.heatmap(
        metric_data,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": primary_metric.upper()},
    )

    ax.set_title(title or f"{primary_metric.upper()} by condition and model", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel("Bias condition")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    out_path = output_dir / f"heatmap_{primary_metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved heatmap to {out_path}")
    return out_path


@task(name="generate_degradation_chart")
def task_generate_degradation_chart(
    pivot: pd.DataFrame,
    primary_metric: str,
    baseline_condition: str,
    output_dir: Path,
    title: str = "",
) -> Path:
    """Generate a grouped bar chart of % degradation vs unbiased baseline.

    Each group of bars is one bias condition; each bar within the group
    is one model.  Positive values always mean "performance got worse."

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    model_cols = [
        c for c in pivot.columns
        if not c.endswith("_deg_pct") and c != "mean_degradation_pct"
    ]
    deg_cols = [f"{c}_deg_pct" for c in model_cols if f"{c}_deg_pct" in pivot.columns]

    if not deg_cols:
        logger.warning("No degradation columns found; skipping degradation chart.")
        out_path = output_dir / f"degradation_{primary_metric}.png"
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No baseline found", ha="center", va="center")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    # Build a tidy DataFrame for plotting
    deg_data = pivot[deg_cols].copy()
    deg_data.columns = model_cols
    # Drop baseline row (degradation is 0 by definition)
    if baseline_condition in deg_data.index:
        deg_data = deg_data.drop(index=baseline_condition)

    # Sort by mean degradation descending (worst at left)
    deg_data = deg_data.loc[deg_data.mean(axis=1).sort_values(ascending=False).index]

    n_conditions = len(deg_data)
    n_models = len(model_cols)
    bar_width = 0.8 / n_models
    x = np.arange(n_conditions)

    fig, ax = plt.subplots(figsize=(max(8, n_conditions * 1.2), 5))

    colors = plt.cm.tab10.colors
    for i, model in enumerate(model_cols):
        offsets = x + (i - n_models / 2 + 0.5) * bar_width
        values = deg_data[model].values
        bars = ax.bar(offsets, values, width=bar_width * 0.9, label=model, color=colors[i % len(colors)])

        # Label bars with their values
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.3 if val >= 0 else -1.5),
                    f"{val:+.1f}%",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(deg_data.index, rotation=30, ha="right")
    ax.set_ylabel(f"Δ {primary_metric.upper()} vs {baseline_condition} (%)")
    ax.set_title(title or f"Performance degradation vs '{baseline_condition}' baseline", pad=14)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()

    out_path = output_dir / f"degradation_{primary_metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved degradation chart to {out_path}")
    return out_path


@task(name="log_comparison_artifacts")
def task_log_artifacts(
    pivot: pd.DataFrame,
    results_df: pd.DataFrame,
    primary_metric: str,
    heatmap_path: Path,
    degradation_path: Path,
    dataset_name: str,
    output_dir: Path,
    wandb_mode: str = "disabled",
) -> None:
    """Save comparison CSV and log all artifacts to W&B.

    Logs:
        - Comparison pivot as a CSV file (always)
        - Heatmap PNG as a W&B Image
        - Degradation chart PNG as a W&B Image
        - Raw results as a W&B Table
    """
    from molgate.tracking import init_run

    # Always save CSV regardless of W&B mode
    csv_path = output_dir / f"comparison_{primary_metric}.csv"
    pivot.to_csv(csv_path)
    logger.info(f"Saved comparison table to {csv_path}")

    raw_csv_path = output_dir / "results_raw.csv"
    results_df.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw results to {raw_csv_path}")

    run = init_run(
        name=f"compare_{dataset_name}",
        config={
            "dataset": dataset_name,
            "primary_metric": primary_metric,
            "n_conditions": len(pivot),
        },
        group=f"compare_{dataset_name}",
        job_type="compare",
        mode=wandb_mode,
    )

    try:
        import wandb

        run.log({
            "heatmap": wandb.Image(str(heatmap_path)),
            "degradation_chart": wandb.Image(str(degradation_path)),
        })

        # Log the pivot as a W&B Table
        model_cols = [
            c for c in pivot.columns
            if not c.endswith("_deg_pct") and c != "mean_degradation_pct"
        ]
        table_df = pivot[model_cols].reset_index()
        if "mean_degradation_pct" in pivot.columns:
            table_df["mean_degradation_pct"] = pivot["mean_degradation_pct"].values

        run.log({"comparison_table": wandb.Table(dataframe=table_df)})
        run.log({"results_table": wandb.Table(dataframe=results_df)})

        logger.info("Logged comparison artifacts to W&B")

    finally:
        run.finish()


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="compare_flow", log_prints=True)
def compare_flow(
    results_df: pd.DataFrame | None = None,
    wandb_group: str | None = None,
    wandb_project: str | None = None,
    dataset_name: str = "solubility",
    primary_metric: str = "rmse",
    baseline_condition: str = "unbiased",
    output_dir: str | Path | None = None,
    wandb_mode: str = "disabled",
) -> dict[str, Any]:
    """Aggregate bias experiment results and generate comparison artifacts.

    Parameters
    ----------
    results_df : pd.DataFrame, optional
        Results table from ``bias_experiment_flow``.  If provided, W&B
        is not queried.
    wandb_group : str, optional
        W&B run group to query (e.g., "bias_experiment_solubility").
        Used only if ``results_df`` is None.
    wandb_project : str, optional
        W&B project name.  Defaults to "molgate" if not provided.
    dataset_name : str
        Dataset name (used for labelling and W&B group naming).
    primary_metric : str
        The metric to pivot on and plot (e.g., "rmse", "auroc").
    baseline_condition : str
        The condition used as the "no-bias" reference for degradation
        computation (default: "unbiased").
    output_dir : str or Path, optional
        Directory to save plots and CSVs.  Defaults to
        ``outputs/compare_<dataset_name>/`` relative to the project root.
    wandb_mode : str
        W&B mode: "online", "offline", or "disabled".

    Returns
    -------
    dict
        Keys: "pivot", "results_df", "heatmap_path", "degradation_path",
        "output_dir".
    """
    # Resolve output directory
    if output_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        out = project_root / "outputs" / f"compare_{dataset_name}"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out}")

    # 1. Load or query results
    loaded_df = task_load_results(
        results_df=results_df,
        wandb_group=wandb_group,
        wandb_project=wandb_project,
        dataset_name=dataset_name,
        primary_metric=primary_metric,
    )

    # 2. Build comparison pivot
    pivot = task_build_comparison_table(
        results_df=loaded_df,
        primary_metric=primary_metric,
        baseline_condition=baseline_condition,
    )

    # 3. Heatmap
    heatmap_path = task_generate_heatmap(
        pivot=pivot,
        primary_metric=primary_metric,
        output_dir=out,
        title=f"{dataset_name.capitalize()} — {primary_metric.upper()} by condition and model",
    )

    # 4. Degradation chart
    degradation_path = task_generate_degradation_chart(
        pivot=pivot,
        primary_metric=primary_metric,
        baseline_condition=baseline_condition,
        output_dir=out,
        title=f"{dataset_name.capitalize()} — degradation vs '{baseline_condition}'",
    )

    # 5. Log artifacts
    task_log_artifacts(
        pivot=pivot,
        results_df=loaded_df,
        primary_metric=primary_metric,
        heatmap_path=heatmap_path,
        degradation_path=degradation_path,
        dataset_name=dataset_name,
        output_dir=out,
        wandb_mode=wandb_mode,
    )

    logger.info(f"\nCompare flow complete. Outputs in: {out}")
    return {
        "pivot": pivot,
        "results_df": loaded_df,
        "heatmap_path": heatmap_path,
        "degradation_path": degradation_path,
        "output_dir": out,
    }


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Build a synthetic results_df that mimics bias_experiment_flow output.
    # This lets us test the whole flow without running actual training.
    rng = np.random.default_rng(42)

    conditions = ["unbiased", "scaffold_top10", "mw_narrow", "target_extremes_removed", "substructure_aromatic"]
    models = ["lgbm_morgan", "lgbm_descriptors", "gnn"]

    # Simulate RMSE values: unbiased ~1.1, biased conditions slightly worse
    degradation_factors = {
        "unbiased": 1.0,
        "scaffold_top10": 1.15,
        "mw_narrow": 1.08,
        "target_extremes_removed": 1.22,
        "substructure_aromatic": 1.30,
    }
    model_base_rmse = {"lgbm_morgan": 1.10, "lgbm_descriptors": 1.05, "gnn": 0.98}

    rows = []
    for condition in conditions:
        for model in models:
            for seed in [42, 123]:
                base = model_base_rmse[model]
                factor = degradation_factors[condition]
                rmse = base * factor + rng.normal(0, 0.02)
                mae = rmse * 0.75 + rng.normal(0, 0.01)
                r2 = 0.85 - (factor - 1.0) * 0.5 + rng.normal(0, 0.01)
                rows.append({
                    "dataset": "solubility",
                    "model": model,
                    "condition": condition,
                    "bias": condition if condition != "unbiased" else "unbiased",
                    "split": "random",
                    "seed": seed,
                    "rmse": max(0.5, rmse),
                    "mae": max(0.3, mae),
                    "r2": min(1.0, r2),
                })

    synthetic_df = pd.DataFrame(rows)

    print("=" * 60)
    print("Compare Flow — synthetic demo (W&B disabled)")
    print("=" * 60)
    print(f"Input: {len(synthetic_df)} rows, {synthetic_df['condition'].nunique()} conditions × "
          f"{synthetic_df['model'].nunique()} models × {synthetic_df['seed'].nunique()} seeds")

    result = compare_flow(
        results_df=synthetic_df,
        dataset_name="solubility",
        primary_metric="rmse",
        baseline_condition="unbiased",
        wandb_mode="disabled",
    )

    print(f"\n{'='*60}")
    print("Comparison pivot (mean RMSE):")
    print(f"{'='*60}")
    model_cols = ["lgbm_morgan", "lgbm_descriptors", "gnn"]
    print(result["pivot"][model_cols].to_string())

    print(f"\n{'='*60}")
    print("Degradation vs unbiased (%):")
    print(f"{'='*60}")
    deg_cols = [f"{m}_deg_pct" for m in model_cols if f"{m}_deg_pct" in result["pivot"].columns]
    if deg_cols:
        print(result["pivot"][deg_cols].rename(columns=lambda c: c.replace("_deg_pct", "")).to_string())

    print(f"\nOutputs written to: {result['output_dir']}")

    import IPython
    IPython.embed()
