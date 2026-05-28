"""Bias diagnostics — measure how a biased dataset differs from the original.

This is the central analysis module for the project's core question: "How does
dataset bias affect molecular property prediction?"  Given an original (clean)
dataset and a biased variant, we quantify the bias across four dimensions:

1. **Descriptor distribution shift** (``descriptor_shift``)
   KS test on each molecular descriptor between original and biased.  Tells us
   *which* chemical properties shifted — did the bias remove heavy molecules?
   Polar molecules?  Flexible molecules?

2. **Scaffold diversity change** (``diversity_change``)
   Compares scaffold ratio, singleton fraction, and top-K coverage before and
   after biasing.  A dataset that lost scaffold diversity will produce models
   that fail on novel chemical series.

3. **Target distribution change** (``target_shift``)
   Compares mean, std, skewness, and range of the target variable.  If bias
   removed extreme values, the target distribution becomes narrower and models
   won't learn the full range.

4. **Adversarial validation** (``adversarial_validation``)
   Train a classifier to distinguish "original" from "biased" molecules using
   Morgan fingerprints.  If the classifier succeeds (high AUROC), the biased
   dataset is structurally distinguishable from the original — confirming the
   bias has a real chemical signature, not just a statistical one.

All four are combined in ``bias_report``, which returns a flat dict suitable
for W&B logging plus DataFrames for notebook display.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from molgate.analysis.descriptors import compare_splits, compute_descriptors
from molgate.analysis.scaffolds import scaffold_diversity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Descriptor distribution shift
# ---------------------------------------------------------------------------

def descriptor_shift(
    smiles_original: list[str],
    smiles_biased: list[str],
) -> pd.DataFrame:
    """Measure descriptor distribution shift between original and biased datasets.

    Wraps ``compare_splits`` with labels appropriate for bias analysis.
    Returns a DataFrame sorted by KS statistic (most shifted first).

    Args:
        smiles_original: SMILES from the full (unbiased) dataset.
        smiles_biased: SMILES from the biased subset.

    Returns:
        DataFrame with columns: descriptor, mean_original, mean_biased,
        mean_diff, ks_statistic, p_value, significant.
    """
    desc_orig = compute_descriptors(smiles_original)
    desc_bias = compute_descriptors(smiles_biased)

    ks_df = compare_splits(
        desc_orig, desc_bias,
        label_a="original", label_b="biased",
    )
    return ks_df


# ---------------------------------------------------------------------------
# 2. Scaffold diversity change
# ---------------------------------------------------------------------------

def diversity_change(
    smiles_original: list[str],
    smiles_biased: list[str],
) -> dict:
    """Compare scaffold diversity between original and biased datasets.

    Args:
        smiles_original: SMILES from the full dataset.
        smiles_biased: SMILES from the biased subset.

    Returns:
        Dict with keys like ``scaffold_ratio_original``, ``scaffold_ratio_biased``,
        ``scaffold_ratio_change`` (biased - original), and similarly for
        singleton_fraction and top_10_coverage.
    """
    div_orig = scaffold_diversity(smiles_original)
    div_bias = scaffold_diversity(smiles_biased)

    result = {
        "n_molecules_original": div_orig["n_molecules"],
        "n_molecules_biased": div_bias["n_molecules"],
        "n_scaffolds_original": div_orig["n_scaffolds"],
        "n_scaffolds_biased": div_bias["n_scaffolds"],
        # Scaffold ratio
        "scaffold_ratio_original": div_orig["scaffold_ratio"],
        "scaffold_ratio_biased": div_bias["scaffold_ratio"],
        "scaffold_ratio_change": div_bias["scaffold_ratio"] - div_orig["scaffold_ratio"],
        # Singleton fraction
        "singleton_fraction_original": div_orig["singleton_fraction"],
        "singleton_fraction_biased": div_bias["singleton_fraction"],
        "singleton_fraction_change": (
            div_bias["singleton_fraction"] - div_orig["singleton_fraction"]
        ),
        # Top-10 coverage (higher = more concentrated = less diverse)
        "top_10_coverage_original": div_orig["top_10_coverage"],
        "top_10_coverage_biased": div_bias["top_10_coverage"],
        "top_10_coverage_change": (
            div_bias["top_10_coverage"] - div_orig["top_10_coverage"]
        ),
    }

    logger.info(
        f"Diversity change: scaffold ratio {result['scaffold_ratio_original']:.3f} → "
        f"{result['scaffold_ratio_biased']:.3f} "
        f"({result['scaffold_ratio_change']:+.3f})"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Target distribution change
# ---------------------------------------------------------------------------

def target_shift(
    target_original: np.ndarray | pd.Series,
    target_biased: np.ndarray | pd.Series,
) -> dict:
    """Compare target variable distributions between original and biased datasets.

    Args:
        target_original: Target values from the full dataset.
        target_biased: Target values from the biased subset.

    Returns:
        Dict with statistical comparisons: mean, std, skewness, min, max,
        range, and KS test results.
    """
    orig = np.asarray(target_original, dtype=float)
    bias = np.asarray(target_biased, dtype=float)

    ks_stat, ks_p = stats.ks_2samp(orig, bias)

    result = {
        "mean_original": float(orig.mean()),
        "mean_biased": float(bias.mean()),
        "mean_change": float(bias.mean() - orig.mean()),
        "std_original": float(orig.std()),
        "std_biased": float(bias.std()),
        "std_change": float(bias.std() - orig.std()),
        "skewness_original": float(stats.skew(orig)),
        "skewness_biased": float(stats.skew(bias)),
        "min_original": float(orig.min()),
        "min_biased": float(bias.min()),
        "max_original": float(orig.max()),
        "max_biased": float(bias.max()),
        "range_original": float(orig.max() - orig.min()),
        "range_biased": float(bias.max() - bias.min()),
        "range_change": float((bias.max() - bias.min()) - (orig.max() - orig.min())),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "target_shift_significant": ks_p < 0.05,
    }

    logger.info(
        f"Target shift: mean {result['mean_original']:.3f} → {result['mean_biased']:.3f} "
        f"({result['mean_change']:+.3f}), "
        f"std {result['std_original']:.3f} → {result['std_biased']:.3f}, "
        f"KS={ks_stat:.4f} (p={ks_p:.2e})"
    )
    return result


# ---------------------------------------------------------------------------
# 4. Adversarial validation
# ---------------------------------------------------------------------------

def adversarial_validation(
    smiles_original: list[str],
    smiles_biased: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Train a classifier to distinguish original from biased molecules.

    Adversarial validation answers: "Is the biased dataset structurally
    distinguishable from the original?"

    We label original molecules as 0 and biased molecules as 1, then train
    a Random Forest on Morgan fingerprints with cross-validation.

    - **AUROC ~ 0.5**: The classifier can't tell them apart → bias doesn't
      have a clear structural signature (e.g., random subsampling).
    - **AUROC ~ 0.8+**: The classifier easily distinguishes them → the bias
      created a structurally distinct subset (e.g., scaffold bias, MW filter).

    Why Random Forest on fingerprints?
      We want a model that's fast, doesn't need tuning, and works well on
      binary fingerprint vectors.  RF with default hyperparameters is a
      reliable baseline for this task.  We avoid neural networks to keep
      this analysis lightweight.

    Args:
        smiles_original: SMILES from the full dataset.
        smiles_biased: SMILES from the biased subset.
        n_folds: Number of cross-validation folds.
        seed: Random seed.

    Returns:
        Dict with keys:
        - ``auroc_mean``: Mean AUROC across folds
        - ``auroc_std``: Std of AUROC across folds
        - ``auroc_folds``: Per-fold AUROC values
        - ``n_original``: Sample count for original
        - ``n_biased``: Sample count for biased
        - ``distinguishable``: True if AUROC > 0.65 (above chance)
    """
    from molgate.data.featurizer import compute_fingerprints

    logger.info(
        f"Adversarial validation: {len(smiles_original)} original vs "
        f"{len(smiles_biased)} biased molecules"
    )

    # Compute fingerprints for both sets
    fps_orig = compute_fingerprints(smiles_original)
    fps_bias = compute_fingerprints(smiles_biased)

    # Stack features and create labels
    X = np.vstack([fps_orig, fps_bias])
    y = np.concatenate([
        np.zeros(len(fps_orig)),
        np.ones(len(fps_bias)),
    ])

    # Train RF with cross-validation
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1,
    )
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="roc_auc")

    result = {
        "auroc_mean": float(scores.mean()),
        "auroc_std": float(scores.std()),
        "auroc_folds": scores.tolist(),
        "n_original": len(smiles_original),
        "n_biased": len(smiles_biased),
        "distinguishable": float(scores.mean()) > 0.65,
    }

    logger.info(
        f"Adversarial AUROC: {result['auroc_mean']:.3f} ± {result['auroc_std']:.3f} "
        f"({'distinguishable' if result['distinguishable'] else 'indistinguishable'})"
    )
    return result


# ---------------------------------------------------------------------------
# 5. Full bias report
# ---------------------------------------------------------------------------

def bias_report(
    smiles_original: list[str],
    smiles_biased: list[str],
    target_original: np.ndarray | pd.Series,
    target_biased: np.ndarray | pd.Series,
    bias_name: str = "unknown",
    run_adversarial: bool = True,
) -> dict:
    """Run all bias diagnostics and return a combined report.

    This is the main entry point for bias analysis.  Call it from Prefect
    flows or notebooks to get a complete picture of how biasing changed
    the dataset.

    Args:
        smiles_original: SMILES from the full dataset.
        smiles_biased: SMILES from the biased subset.
        target_original: Target values for the full dataset.
        target_biased: Target values for the biased subset.
        bias_name: Name of the bias condition (for logging/W&B tags).
        run_adversarial: Whether to run adversarial validation (slower).

    Returns:
        Dict with:
        - ``"descriptor_shift"``: KS comparison DataFrame
        - ``"diversity_change"``: diversity metrics dict
        - ``"target_shift"``: target comparison dict
        - ``"adversarial"``: adversarial validation dict (if run_adversarial)
        - Scalar summaries for direct W&B logging
    """
    logger.info(
        f"Bias report for '{bias_name}': "
        f"{len(smiles_original)} original → {len(smiles_biased)} biased"
    )

    # Descriptor shift
    desc_shift = descriptor_shift(smiles_original, smiles_biased)
    n_sig_descriptors = int(desc_shift["significant"].sum())

    # Diversity change
    div_change = diversity_change(smiles_original, smiles_biased)

    # Target shift
    tgt_shift = target_shift(target_original, target_biased)

    report = {
        "bias_name": bias_name,
        "n_original": len(smiles_original),
        "n_biased": len(smiles_biased),
        "retention_rate": len(smiles_biased) / len(smiles_original),
        # Descriptor shift
        "descriptor_shift": desc_shift,
        "n_significant_descriptors": n_sig_descriptors,
        "max_ks_descriptor": desc_shift.iloc[0]["descriptor"],
        "max_ks_statistic": float(desc_shift.iloc[0]["ks_statistic"]),
        # Diversity
        "diversity_change": div_change,
        "scaffold_ratio_change": div_change["scaffold_ratio_change"],
        # Target
        "target_shift": tgt_shift,
        "target_mean_change": tgt_shift["mean_change"],
        "target_std_change": tgt_shift["std_change"],
        "target_ks_statistic": tgt_shift["ks_statistic"],
    }

    # Adversarial validation (optional — adds ~10-20 seconds)
    if run_adversarial:
        adv = adversarial_validation(smiles_original, smiles_biased)
        report["adversarial"] = adv
        report["adversarial_auroc"] = adv["auroc_mean"]
        report["adversarial_distinguishable"] = adv["distinguishable"]

    return report


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.bias import bias_by_scaffold, bias_by_target_region
    from molgate.data.loaders import load_dataset

    df = load_dataset("solubility")
    smiles_orig = df["smiles"].tolist()
    target_orig = df["y"].values

    # Test with scaffold bias
    biased_df, meta = bias_by_scaffold(df, top_n=10)
    smiles_biased = biased_df["smiles"].tolist()
    target_biased = biased_df["y"].values

    print("\n" + "=" * 60)
    print("BIAS REPORT: Scaffold bias (top 10)")
    print("=" * 60)

    report = bias_report(
        smiles_orig, smiles_biased,
        target_orig, target_biased,
        bias_name="scaffold_top10",
    )

    print(f"\n  Retention: {report['retention_rate']:.1%}")
    print(f"  Significant descriptor shifts: {report['n_significant_descriptors']}/12")
    print(f"  Most shifted: {report['max_ks_descriptor']} (KS={report['max_ks_statistic']:.4f})")
    print(f"  Scaffold ratio change: {report['scaffold_ratio_change']:+.3f}")
    print(f"  Target mean change: {report['target_mean_change']:+.3f}")
    print(f"  Target std change: {report['target_std_change']:+.3f}")
    if "adversarial_auroc" in report:
        print(f"  Adversarial AUROC: {report['adversarial_auroc']:.3f}")

    # Also test with target region bias for comparison
    biased_df2, meta2 = bias_by_target_region(df, quantile_low=0.1, quantile_high=0.9)

    print("\n" + "=" * 60)
    print("BIAS REPORT: Target region bias (drop extremes)")
    print("=" * 60)

    report2 = bias_report(
        smiles_orig, biased_df2["smiles"].tolist(),
        target_orig, biased_df2["y"].values,
        bias_name="target_region_0.1_0.9",
    )

    print(f"\n  Retention: {report2['retention_rate']:.1%}")
    print(f"  Significant descriptor shifts: {report2['n_significant_descriptors']}/12")
    print(f"  Target mean change: {report2['target_mean_change']:+.3f}")
    print(f"  Target std change: {report2['target_std_change']:+.3f}")
    if "adversarial_auroc" in report2:
        print(f"  Adversarial AUROC: {report2['adversarial_auroc']:.3f}")

    import IPython; IPython.embed()
