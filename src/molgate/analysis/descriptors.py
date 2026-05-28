"""Descriptor-level analysis — distribution statistics, split comparisons, correlations.

This module provides functions for analyzing molecular descriptor distributions
across datasets and splits.  It builds on ``molgate.data.featurizer.compute_descriptors``
which computes the raw descriptor values; this module adds the *statistical* layer:

1. **Summary statistics** (``descriptor_summary``)
   Per-descriptor mean, std, min, max, median, skewness, kurtosis.
   Gives you a quick fingerprint of a dataset's chemical space.

2. **Train/test distribution comparison** (``compare_splits``)
   For each descriptor, runs a two-sample Kolmogorov-Smirnov test between
   two splits (e.g., train vs test).  The KS statistic measures the maximum
   difference between the two empirical CDFs — a value near 0 means the
   distributions are indistinguishable, near 1 means completely different.
   This is the standard non-parametric test for distribution shift.

3. **Descriptor-target correlations** (``descriptor_target_correlations``)
   Pearson and Spearman correlation between each descriptor and the target.
   Pearson captures linear relationships, Spearman captures monotonic
   (including non-linear) relationships.  High |correlation| means the
   descriptor is informative for prediction.

4. **Full analysis report** (``descriptor_analysis``)
   Convenience function that runs all three analyses and returns a single
   dict suitable for W&B logging or JSON serialization.

Why is this in ``analysis/`` and not ``data/``?
  The ``data/`` module handles loading and featurization (transforming
  molecules into numbers).  ``analysis/`` handles statistical characterization
  (what do those numbers tell us about the dataset).  Keeping them separate
  means the data layer has no dependency on scipy.stats.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from molgate.data.featurizer import DESCRIPTOR_LIST, compute_descriptors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Summary statistics
# ---------------------------------------------------------------------------

def descriptor_summary(desc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each descriptor.

    Args:
        desc_df: DataFrame of descriptor values, as returned by
            ``compute_descriptors``.  Columns are descriptor names,
            rows are molecules.

    Returns:
        DataFrame with one row per descriptor and columns:
        mean, std, min, q25, median, q75, max, skewness, kurtosis.

    Example::

        from molgate.data.featurizer import compute_descriptors
        from molgate.analysis.descriptors import descriptor_summary

        desc = compute_descriptors(df["smiles"].tolist())
        summary = descriptor_summary(desc)
        print(summary)
    """
    records = []
    for col in desc_df.columns:
        values = desc_df[col].dropna()
        records.append({
            "descriptor": col,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "q25": values.quantile(0.25),
            "median": values.median(),
            "q75": values.quantile(0.75),
            "max": values.max(),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
        })

    result = pd.DataFrame(records).set_index("descriptor")
    logger.info(f"Summary statistics for {len(result)} descriptors")
    return result


# ---------------------------------------------------------------------------
# 2. Train/test distribution comparison (KS test)
# ---------------------------------------------------------------------------

def compare_splits(
    desc_a: pd.DataFrame,
    desc_b: pd.DataFrame,
    label_a: str = "train",
    label_b: str = "test",
) -> pd.DataFrame:
    """Compare descriptor distributions between two splits using the KS test.

    The two-sample Kolmogorov-Smirnov test asks: "Could these two samples
    have been drawn from the same underlying distribution?"

    - **KS statistic**: Maximum absolute difference between the two empirical
      CDFs.  Ranges from 0 (identical distributions) to 1 (no overlap).
    - **p-value**: Probability of seeing a KS statistic this large if the
      two samples *were* from the same distribution.  Small p-value (< 0.05)
      means the distributions are significantly different.

    Why KS and not t-test?
      The t-test only compares means and assumes normality.  KS compares
      the entire distribution shape — it catches shifts in spread, skewness,
      and tails that a t-test would miss.  Most molecular descriptors are
      non-normal (e.g., MW is right-skewed), so KS is more appropriate.

    Args:
        desc_a: Descriptor DataFrame for split A (e.g., training set).
        desc_b: Descriptor DataFrame for split B (e.g., test set).
        label_a: Label for split A (used in column names).
        label_b: Label for split B (used in column names).

    Returns:
        DataFrame with one row per descriptor and columns:
        descriptor, mean_a, mean_b, mean_diff, ks_statistic, p_value, significant.
        Sorted by ks_statistic descending (most shifted descriptors first).
    """
    if set(desc_a.columns) != set(desc_b.columns):
        raise ValueError(
            f"Descriptor columns don't match: "
            f"{set(desc_a.columns) - set(desc_b.columns)} in A but not B, "
            f"{set(desc_b.columns) - set(desc_a.columns)} in B but not A"
        )

    records = []
    for col in desc_a.columns:
        vals_a = desc_a[col].dropna().values
        vals_b = desc_b[col].dropna().values

        ks_stat, p_val = stats.ks_2samp(vals_a, vals_b)

        records.append({
            "descriptor": col,
            f"mean_{label_a}": vals_a.mean(),
            f"mean_{label_b}": vals_b.mean(),
            "mean_diff": abs(vals_a.mean() - vals_b.mean()),
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
        })

    result = pd.DataFrame(records).sort_values("ks_statistic", ascending=False)
    result = result.reset_index(drop=True)

    n_sig = result["significant"].sum()
    logger.info(
        f"KS comparison ({label_a} vs {label_b}): "
        f"{n_sig}/{len(result)} descriptors show significant shift (p < 0.05)"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Descriptor-target correlations
# ---------------------------------------------------------------------------

def descriptor_target_correlations(
    desc_df: pd.DataFrame,
    target: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations between descriptors and target.

    Pearson correlation measures *linear* association: r = 1 means perfect
    positive linear relationship, r = -1 means perfect negative linear,
    r = 0 means no linear relationship (but could still be non-linearly
    related).

    Spearman correlation measures *monotonic* association.  It's computed
    on ranks rather than raw values, so it captures any monotonic relationship
    (linear, logarithmic, exponential, etc.).  More robust to outliers.

    For ADMET prediction, descriptors with high |Spearman| are likely to be
    useful features for tree-based models (which learn monotonic splits).
    Descriptors with high |Pearson| are useful for linear models.

    Args:
        desc_df: Descriptor DataFrame (from ``compute_descriptors``).
        target: Target values, same length as desc_df.

    Returns:
        DataFrame with columns: descriptor, pearson_r, pearson_p,
        spearman_r, spearman_p, abs_pearson, abs_spearman.
        Sorted by abs_spearman descending (most correlated first).
    """
    target = np.asarray(target)
    if len(target) != len(desc_df):
        raise ValueError(
            f"Length mismatch: {len(desc_df)} descriptors vs {len(target)} targets"
        )

    records = []
    for col in desc_df.columns:
        values = desc_df[col].values
        # Drop NaNs pairwise (both descriptor and target must be valid)
        mask = ~(np.isnan(values) | np.isnan(target))
        v = values[mask]
        t = target[mask]

        if len(v) < 3:
            # Not enough data for correlation
            records.append({
                "descriptor": col,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "abs_pearson": np.nan,
                "abs_spearman": np.nan,
            })
            continue

        pr, pp = stats.pearsonr(v, t)
        sr, sp = stats.spearmanr(v, t)

        records.append({
            "descriptor": col,
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_r": sr,
            "spearman_p": sp,
            "abs_pearson": abs(pr),
            "abs_spearman": abs(sr),
        })

    result = pd.DataFrame(records).sort_values("abs_spearman", ascending=False)
    result = result.reset_index(drop=True)

    # Log top 3 most correlated
    top3 = result.head(3)
    top_str = ", ".join(
        f"{row['descriptor']} (r={row['spearman_r']:.3f})"
        for _, row in top3.iterrows()
    )
    logger.info(f"Top correlated descriptors with target: {top_str}")
    return result


# ---------------------------------------------------------------------------
# 4. Full analysis report
# ---------------------------------------------------------------------------

def descriptor_analysis(
    smiles_a: list[str],
    smiles_b: list[str] | None = None,
    target_a: np.ndarray | pd.Series | None = None,
    label_a: str = "train",
    label_b: str = "test",
) -> dict:
    """Run full descriptor analysis and return a W&B-loggable report.

    This is the convenience entry point that combines all three analyses.
    Use this from Prefect flows or notebooks when you want everything at once.

    Args:
        smiles_a: SMILES for the primary split (e.g., training set).
        smiles_b: SMILES for the comparison split (e.g., test set).
            If None, skips the KS comparison.
        target_a: Target values for split A.  If provided, computes
            descriptor-target correlations.
        label_a: Label for split A.
        label_b: Label for split B.

    Returns:
        Dict with keys:
        - ``"summary"``: Summary statistics DataFrame
        - ``"ks_comparison"``: KS test results DataFrame (if smiles_b given)
        - ``"correlations"``: Correlation DataFrame (if target_a given)
        - ``"n_molecules_a"``: Count for split A
        - ``"n_molecules_b"``: Count for split B (if given)
        - ``"n_significant_ks"``: Number of descriptors with p < 0.05
        - ``"max_ks_descriptor"``: Descriptor with highest KS statistic
        - ``"max_ks_statistic"``: The KS statistic value
        - ``"top_correlated_descriptor"``: Most correlated with target
        - ``"top_correlation"``: The Spearman r value
    """
    logger.info(f"Running descriptor analysis ({len(smiles_a)} molecules in {label_a})")

    desc_a = compute_descriptors(smiles_a)
    summary = descriptor_summary(desc_a)

    report: dict = {
        "summary": summary,
        "n_molecules_a": len(smiles_a),
    }

    # KS comparison if a second split is provided
    if smiles_b is not None:
        desc_b = compute_descriptors(smiles_b)
        ks_df = compare_splits(desc_a, desc_b, label_a, label_b)
        report["ks_comparison"] = ks_df
        report["n_molecules_b"] = len(smiles_b)
        report["n_significant_ks"] = int(ks_df["significant"].sum())
        report["max_ks_descriptor"] = ks_df.iloc[0]["descriptor"]
        report["max_ks_statistic"] = float(ks_df.iloc[0]["ks_statistic"])

    # Correlations if target is provided
    if target_a is not None:
        corr_df = descriptor_target_correlations(desc_a, target_a)
        report["correlations"] = corr_df
        report["top_correlated_descriptor"] = corr_df.iloc[0]["descriptor"]
        report["top_correlation"] = float(corr_df.iloc[0]["spearman_r"])

    return report


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.loaders import load_dataset
    from molgate.data.splits import random_split

    # Load solubility and split
    df = load_dataset("solubility")
    splits = random_split(df, seed=42)

    train_df = splits["train"]
    test_df = splits["test"]

    # Run full analysis
    report = descriptor_analysis(
        smiles_a=train_df["smiles"].tolist(),
        smiles_b=test_df["smiles"].tolist(),
        target_a=train_df["y"].values,
        label_a="train",
        label_b="test",
    )

    print("\n=== Summary Statistics ===")
    print(report["summary"].to_string())

    print("\n=== KS Comparison (train vs test) ===")
    ks = report["ks_comparison"]
    print(ks[["descriptor", "ks_statistic", "p_value", "significant"]].to_string(index=False))
    print(f"\n  Significant shifts: {report['n_significant_ks']}/{len(ks)}")
    print(f"  Largest shift: {report['max_ks_descriptor']} (KS={report['max_ks_statistic']:.4f})")

    print("\n=== Descriptor-Target Correlations ===")
    corr = report["correlations"]
    print(corr[["descriptor", "pearson_r", "spearman_r"]].to_string(index=False))
    print(f"\n  Top correlated: {report['top_correlated_descriptor']} (r={report['top_correlation']:.4f})")

    import IPython; IPython.embed()
