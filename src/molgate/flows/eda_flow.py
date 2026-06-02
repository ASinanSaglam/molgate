"""Prefect flow: Exploratory Data Analysis for a single dataset.

This flow runs a comprehensive EDA pipeline for any TDC ADMET dataset.
It orchestrates the analysis modules we built in Phase 2 and logs
everything to W&B for interactive exploration.

Tasks in the flow:
    1. Load and clean the dataset
    2. Split into train/test (for comparing distributions)
    3. Compute descriptor statistics
    4. Analyse target distribution
    5. Scaffold analysis (diversity, frequency, overlap)
    6. Activity cliff detection
    7. Adversarial validation (train vs test structural similarity)
    8. Log all results to W&B

Each step is a Prefect ``@task`` — this gives us:
    - Per-task retry logic (network errors on TDC download, etc.)
    - Granular logging in the Prefect UI
    - Cached results on re-runs (via Prefect's caching)
    - Clear dependency graph between steps

The flow is idempotent: running it twice produces the same results.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from prefect import flow, task

from molgate.data.featurizer import compute_descriptors
from molgate.data.loaders import load_dataset
from molgate.data.registry import get_dataset_info
from molgate.data.splits import random_split, scaffold_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load_dataset", retries=2, retry_delay_seconds=10)
def task_load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load and clean a TDC ADMET dataset.

    Retries twice with 10s delay — TDC downloads can be flaky.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    df = load_dataset(dataset_name)
    logger.info(f"  Loaded {len(df)} molecules")
    return df


@task(name="split_dataset")
def task_split_dataset(
    df: pd.DataFrame,
    split_type: str = "random",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split dataset into train/val/test."""
    if split_type == "scaffold":
        splits = scaffold_split(df, seed=seed)
    else:
        splits = random_split(df, seed=seed)

    for name, split_df in splits.items():
        logger.info(f"  {name}: {len(split_df)} molecules")
    return splits


@task(name="compute_target_stats")
def task_target_stats(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Compute target distribution statistics."""
    y = df["y"].values
    stats = {
        "dataset": dataset_name,
        "n_molecules": len(df),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
        "target_median": float(np.median(y)),
        "target_min": float(np.min(y)),
        "target_max": float(np.max(y)),
        "target_range": float(np.ptp(y)),
        "target_skewness": float(pd.Series(y).skew()),
        "target_kurtosis": float(pd.Series(y).kurtosis()),
    }
    logger.info(
        f"  Target stats: mean={stats['target_mean']:.3f}, "
        f"std={stats['target_std']:.3f}, range=[{stats['target_min']:.3f}, {stats['target_max']:.3f}]"
    )
    return stats


@task(name="descriptor_analysis")
def task_descriptor_analysis(
    train_smiles: list[str],
    test_smiles: list[str],
    train_targets: np.ndarray,
) -> dict[str, Any]:
    """Compute descriptor statistics, split comparison, and target correlations."""
    from molgate.analysis.descriptors import descriptor_analysis

    report = descriptor_analysis(
        smiles_a=train_smiles,
        smiles_b=test_smiles,
        target_a=train_targets,
        label_a="train",
        label_b="test",
    )
    logger.info(
        f"  Descriptor analysis: {len(report.get('summary', []))} descriptors, "
        f"top corr={report.get('correlations', pd.DataFrame()).iloc[0]['abs_spearman']:.3f}"
        if len(report.get("correlations", pd.DataFrame())) > 0
        else "  Descriptor analysis complete"
    )
    return report


@task(name="scaffold_analysis")
def task_scaffold_analysis(
    train_smiles: list[str],
    test_smiles: list[str],
    train_targets: np.ndarray,
) -> dict[str, Any]:
    """Scaffold diversity, frequency, overlap, and per-scaffold target stats."""
    from molgate.analysis.scaffolds import scaffold_analysis

    report = scaffold_analysis(
        smiles_a=train_smiles,
        smiles_b=test_smiles,
        target_a=train_targets,
        label_a="train",
        label_b="test",
    )
    diversity = report.get("diversity_a", {})
    logger.info(
        f"  Scaffold analysis: ratio={diversity.get('scaffold_ratio', 0):.3f}, "
        f"singletons={diversity.get('singleton_fraction', 0):.3f}"
    )
    return report


@task(name="cliff_analysis")
def task_cliff_analysis(
    smiles_list: list[str],
    targets: np.ndarray,
) -> dict[str, Any]:
    """Activity cliff detection."""
    from molgate.analysis.cliffs import cliff_analysis

    report = cliff_analysis(
        smiles_list=smiles_list,
        target=targets,
        sim_threshold=0.7,
    )
    summary = report.get("summary", {})
    logger.info(
        f"  Cliff analysis: {summary.get('n_cliffs', 0)} cliffs, "
        f"fraction={summary.get('cliff_fraction', 0):.3f}"
    )
    return report


@task(name="adversarial_validation")
def task_adversarial_validation(
    train_smiles: list[str],
    test_smiles: list[str],
) -> dict[str, Any]:
    """Train vs test adversarial validation (RF on Morgan FPs)."""
    from molgate.analysis.bias_diagnostics import adversarial_validation

    result = adversarial_validation(
        smiles_original=train_smiles,
        smiles_biased=test_smiles,
    )
    logger.info(
        f"  Adversarial validation: AUROC={result.get('auroc_mean', 0):.3f} "
        f"+/- {result.get('auroc_std', 0):.3f}"
    )
    return result


@task(name="log_eda_to_wandb")
def task_log_eda_to_wandb(
    dataset_name: str,
    target_stats: dict,
    descriptor_report: dict,
    scaffold_report: dict,
    cliff_report: dict,
    adversarial_result: dict,
    wandb_mode: str = "disabled",
) -> None:
    """Log all EDA results to a W&B run."""
    from molgate.tracking import build_tags, init_run

    run = init_run(
        name=f"eda_{dataset_name}",
        config={
            "dataset": dataset_name,
            "flow": "eda",
            **target_stats,
        },
        tags=build_tags(dataset=dataset_name, extra=["eda"]),
        group=f"eda_{dataset_name}",
        job_type="eda",
        mode=wandb_mode,
    )

    try:
        # Target distribution stats
        run.summary.update(target_stats)

        # Scaffold diversity
        diversity = scaffold_report.get("diversity_a", {})
        run.summary.update({
            f"scaffold_{k}": v for k, v in diversity.items()
            if isinstance(v, (int, float))
        })

        # Scaffold overlap with test
        overlap = scaffold_report.get("overlap", {})
        if overlap:
            run.summary.update({
                f"scaffold_overlap_{k}": v for k, v in overlap.items()
                if isinstance(v, (int, float))
            })

        # Cliff summary
        cliff_summary = cliff_report.get("summary", {})
        run.summary.update({
            f"cliff_{k}": v for k, v in cliff_summary.items()
            if isinstance(v, (int, float))
        })

        # Adversarial validation
        run.summary.update({
            f"adversarial_{k}": v for k, v in adversarial_result.items()
            if isinstance(v, (int, float))
        })

        # Log descriptor summary as a W&B Table
        import wandb
        desc_summary = descriptor_report.get("summary")
        if desc_summary is not None and isinstance(desc_summary, pd.DataFrame):
            run.log({"descriptor_summary": wandb.Table(dataframe=desc_summary)})

        # Log scaffold frequency table
        freq_table = scaffold_report.get("frequency_table_a")
        if freq_table is not None and isinstance(freq_table, pd.DataFrame):
            run.log({"scaffold_frequency": wandb.Table(dataframe=freq_table.head(20))})

        logger.info(f"  Logged EDA results to W&B run: {run.name}")

    finally:
        run.finish()


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="eda_flow", log_prints=True)
def eda_flow(
    dataset_name: str = "solubility",
    split_type: str = "random",
    seed: int = 42,
    wandb_mode: str = "disabled",
) -> dict[str, Any]:
    """Run full EDA for a single dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (must be in the registry).
    split_type : str
        "random" or "scaffold" — how to split for train/test comparison.
    seed : int
        Random seed for splitting.
    wandb_mode : str
        W&B mode: "online", "offline", or "disabled".

    Returns
    -------
    dict
        EDA summary containing target stats, descriptor report,
        scaffold report, cliff report, and adversarial validation.
    """
    logger.info(f"Starting EDA flow for {dataset_name!r}")

    # 1. Load dataset
    df = task_load_dataset(dataset_name)

    # 2. Split for distribution comparison
    splits = task_split_dataset(df, split_type=split_type, seed=seed)
    train_df = splits["train"]
    test_df = splits["test"]

    train_smiles = train_df["smiles"].tolist()
    test_smiles = test_df["smiles"].tolist()
    train_targets = train_df["y"].values

    # 3. Target distribution stats
    target_stats = task_target_stats(df, dataset_name)

    # 4. Descriptor analysis (train vs test comparison + target correlations)
    descriptor_report = task_descriptor_analysis(
        train_smiles, test_smiles, train_targets,
    )

    # 5. Scaffold analysis
    scaffold_report = task_scaffold_analysis(
        train_smiles, test_smiles, train_targets,
    )

    # 6. Activity cliff detection (on full dataset)
    cliff_report = task_cliff_analysis(
        df["smiles"].tolist(), df["y"].values,
    )

    # 7. Adversarial validation (train vs test)
    adversarial_result = task_adversarial_validation(
        train_smiles, test_smiles,
    )

    # 8. Log everything to W&B
    task_log_eda_to_wandb(
        dataset_name=dataset_name,
        target_stats=target_stats,
        descriptor_report=descriptor_report,
        scaffold_report=scaffold_report,
        cliff_report=cliff_report,
        adversarial_result=adversarial_result,
        wandb_mode=wandb_mode,
    )

    # Return combined summary
    summary = {
        "dataset": dataset_name,
        "n_molecules": len(df),
        "split_type": split_type,
        "target_stats": target_stats,
        "descriptor_report": descriptor_report,
        "scaffold_report": scaffold_report,
        "cliff_report": cliff_report,
        "adversarial_validation": adversarial_result,
    }

    logger.info(f"EDA flow complete for {dataset_name!r}")
    return summary


# ---------------------------------------------------------------------------
# Demo / interactive testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Running EDA flow for solubility (disabled W&B)")
    print("=" * 60)

    result = eda_flow(
        dataset_name="solubility",
        split_type="random",
        seed=42,
        wandb_mode="disabled",
    )

    print(f"\n{'=' * 60}")
    print("EDA Summary")
    print(f"{'=' * 60}")

    # Target stats
    ts = result["target_stats"]
    print(f"\nDataset: {ts['dataset']} ({ts['n_molecules']} molecules)")
    print(f"  Target: mean={ts['target_mean']:.3f}, std={ts['target_std']:.3f}")
    print(f"  Range: [{ts['target_min']:.3f}, {ts['target_max']:.3f}]")
    print(f"  Skewness: {ts['target_skewness']:.3f}, Kurtosis: {ts['target_kurtosis']:.3f}")

    # Scaffold diversity
    div = result["scaffold_report"].get("diversity_a", {})
    print(f"\nScaffold diversity (train):")
    print(f"  Ratio: {div.get('scaffold_ratio', 0):.3f}")
    print(f"  Singletons: {div.get('singleton_fraction', 0):.3f}")
    print(f"  Top-10 coverage: {div.get('top_10_coverage', 0):.3f}")

    # Cliffs
    cs = result["cliff_report"].get("summary", {})
    print(f"\nActivity cliffs:")
    print(f"  N cliffs: {cs.get('n_cliffs', 0)}")
    print(f"  Cliff fraction: {cs.get('cliff_fraction', 0):.3f}")
    print(f"  Molecules involved: {cs.get('n_cliff_molecules', 0)}")

    # Adversarial validation
    av = result["adversarial_validation"]
    print(f"\nAdversarial validation (train vs test):")
    print(f"  AUROC: {av.get('auroc_mean', 0):.3f} +/- {av.get('auroc_std', 0):.3f}")

    import IPython
    IPython.embed()