"""Scaffold-level analysis — diversity metrics, frequency distributions, split overlap.

Scaffolds (Bemis-Murcko frameworks) are the core ring systems of molecules.
Molecules that share a scaffold belong to the same chemical series and
tend to have correlated properties.  Scaffold analysis tells us:

1. **How diverse is a dataset?**
   A dataset with 2000 molecules but only 50 unique scaffolds is much less
   diverse than one with 500 unique scaffolds.  Low diversity means models
   may not generalise to novel chemical series.

2. **How concentrated is the dataset?**
   If the top 10 scaffolds account for 80% of molecules, the dataset is
   dominated by a few chemical series.  Models trained on this will be
   biased toward those series.

3. **Do train and test share scaffolds?**
   Scaffold overlap between splits is a measure of information leakage.
   High overlap means the test set isn't truly evaluating generalisation.
   A scaffold split by definition has zero overlap; a random split may
   have 80%+ overlap.

Key metrics implemented:
  - **Scaffold ratio**: unique_scaffolds / n_molecules (0 to 1, higher = more diverse)
  - **Singleton fraction**: scaffolds appearing exactly once / total scaffolds
  - **Top-K coverage**: fraction of molecules covered by the K most common scaffolds
  - **Scaffold overlap**: Jaccard index of scaffold sets between two splits
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaffold extraction
# ---------------------------------------------------------------------------

def get_scaffold(smiles: str) -> str | None:
    """Extract the Bemis-Murcko scaffold from a SMILES string.

    Returns the canonical SMILES of the core ring system.  Side chains
    are stripped, but linker atoms between rings are preserved.

    Args:
        smiles: Canonical SMILES string.

    Returns:
        Scaffold SMILES, or None for acyclic molecules (no rings)
        or unparseable SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    # Empty scaffold means acyclic molecule
    if not scaffold_smiles:
        return None
    return scaffold_smiles


def get_scaffolds(smiles_list: list[str]) -> list[str | None]:
    """Extract scaffolds for a list of SMILES.

    Args:
        smiles_list: List of canonical SMILES.

    Returns:
        List of scaffold SMILES (same length as input).
        None for acyclic or unparseable molecules.
    """
    scaffolds = [get_scaffold(s) for s in smiles_list]
    n_valid = sum(1 for s in scaffolds if s is not None)
    n_acyclic = len(scaffolds) - n_valid
    logger.info(
        f"Extracted scaffolds: {n_valid} with rings, {n_acyclic} acyclic/invalid"
    )
    return scaffolds


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def scaffold_diversity(smiles_list: list[str]) -> dict:
    """Compute scaffold diversity metrics for a dataset.

    Args:
        smiles_list: List of canonical SMILES.

    Returns:
        Dict with keys:
        - ``n_molecules``: Total molecule count
        - ``n_scaffolds``: Number of unique scaffolds (excluding acyclics)
        - ``n_acyclic``: Number of acyclic molecules
        - ``scaffold_ratio``: unique_scaffolds / n_molecules (higher = more diverse)
        - ``singleton_fraction``: fraction of scaffolds appearing exactly once
        - ``top_1_coverage``: fraction of molecules in the most common scaffold
        - ``top_5_coverage``: fraction covered by top 5 scaffolds
        - ``top_10_coverage``: fraction covered by top 10 scaffolds
        - ``top_20_coverage``: fraction covered by top 20 scaffolds
        - ``scaffold_counts``: Counter of scaffold → count (for further analysis)
    """
    scaffolds = get_scaffolds(smiles_list)
    n_total = len(smiles_list)

    # Separate acyclic from ring-containing
    ring_scaffolds = [s for s in scaffolds if s is not None]
    n_acyclic = n_total - len(ring_scaffolds)

    # Count scaffold frequencies
    counts = Counter(ring_scaffolds)
    n_unique = len(counts)

    # Singleton fraction: scaffolds that appear exactly once
    n_singletons = sum(1 for c in counts.values() if c == 1)
    singleton_frac = n_singletons / n_unique if n_unique > 0 else 0.0

    # Top-K coverage: fraction of molecules in the K most common scaffolds
    sorted_counts = counts.most_common()

    def top_k_coverage(k: int) -> float:
        top_k_total = sum(c for _, c in sorted_counts[:k])
        return top_k_total / n_total if n_total > 0 else 0.0

    result = {
        "n_molecules": n_total,
        "n_scaffolds": n_unique,
        "n_acyclic": n_acyclic,
        "scaffold_ratio": n_unique / n_total if n_total > 0 else 0.0,
        "singleton_fraction": singleton_frac,
        "top_1_coverage": top_k_coverage(1),
        "top_5_coverage": top_k_coverage(5),
        "top_10_coverage": top_k_coverage(10),
        "top_20_coverage": top_k_coverage(20),
        "scaffold_counts": counts,
    }

    logger.info(
        f"Scaffold diversity: {n_unique} unique scaffolds from {n_total} molecules "
        f"(ratio={result['scaffold_ratio']:.3f}, singletons={singleton_frac:.1%}, "
        f"top-10 coverage={result['top_10_coverage']:.1%})"
    )
    return result


# ---------------------------------------------------------------------------
# Scaffold frequency table
# ---------------------------------------------------------------------------

def scaffold_frequency_table(
    smiles_list: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a frequency table of the most common scaffolds.

    Useful for notebooks — shows which chemical series dominate the dataset.

    Args:
        smiles_list: List of canonical SMILES.
        top_n: Number of top scaffolds to include.

    Returns:
        DataFrame with columns: scaffold, count, fraction, cumulative_fraction.
        Sorted by count descending.
    """
    scaffolds = get_scaffolds(smiles_list)
    ring_scaffolds = [s for s in scaffolds if s is not None]
    counts = Counter(ring_scaffolds)
    n_total = len(smiles_list)

    rows = []
    cumulative = 0
    for scaffold, count in counts.most_common(top_n):
        frac = count / n_total
        cumulative += frac
        rows.append({
            "scaffold": scaffold,
            "count": count,
            "fraction": frac,
            "cumulative_fraction": cumulative,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split overlap analysis
# ---------------------------------------------------------------------------

def scaffold_overlap(
    smiles_a: list[str],
    smiles_b: list[str],
    label_a: str = "train",
    label_b: str = "test",
) -> dict:
    """Measure scaffold overlap between two splits.

    Overlap is measured by the Jaccard index: |A ∩ B| / |A ∪ B|.
    A Jaccard of 0 means no shared scaffolds (perfect scaffold split).
    A Jaccard of 1 means identical scaffold sets.

    Also reports what fraction of each split's molecules belong to
    shared scaffolds — this captures the *impact* of the overlap
    (sharing 1 singleton scaffold matters less than sharing the
    most common scaffold).

    Args:
        smiles_a: SMILES for split A.
        smiles_b: SMILES for split B.
        label_a: Label for split A.
        label_b: Label for split B.

    Returns:
        Dict with keys:
        - ``n_scaffolds_a``, ``n_scaffolds_b``: unique scaffold counts
        - ``n_shared``: scaffolds appearing in both splits
        - ``n_union``: total unique scaffolds across both
        - ``jaccard``: Jaccard index (shared / union)
        - ``frac_a_in_shared``: fraction of A's molecules whose scaffold is shared
        - ``frac_b_in_shared``: fraction of B's molecules whose scaffold is shared
    """
    scaffolds_a = get_scaffolds(smiles_a)
    scaffolds_b = get_scaffolds(smiles_b)

    # Unique scaffold sets (exclude None / acyclic)
    set_a = {s for s in scaffolds_a if s is not None}
    set_b = {s for s in scaffolds_b if s is not None}

    shared = set_a & set_b
    union = set_a | set_b

    jaccard = len(shared) / len(union) if union else 0.0

    # Fraction of molecules in each split that belong to shared scaffolds
    n_a_shared = sum(1 for s in scaffolds_a if s in shared)
    n_b_shared = sum(1 for s in scaffolds_b if s in shared)

    result = {
        "n_scaffolds_a": len(set_a),
        "n_scaffolds_b": len(set_b),
        "n_shared": len(shared),
        "n_union": len(union),
        "jaccard": jaccard,
        f"frac_{label_a}_in_shared": n_a_shared / len(smiles_a) if smiles_a else 0.0,
        f"frac_{label_b}_in_shared": n_b_shared / len(smiles_b) if smiles_b else 0.0,
    }

    logger.info(
        f"Scaffold overlap ({label_a} vs {label_b}): "
        f"{len(shared)} shared / {len(union)} total (Jaccard={jaccard:.3f})"
    )
    return result


# ---------------------------------------------------------------------------
# Per-scaffold target statistics
# ---------------------------------------------------------------------------

def scaffold_target_stats(
    smiles_list: list[str],
    target: np.ndarray | pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute target distribution statistics per scaffold.

    Shows whether different scaffolds have different property distributions.
    Large differences indicate that scaffold identity is predictive of the
    target — meaning a scaffold split will create a harder test than a
    random split.

    Args:
        smiles_list: List of canonical SMILES.
        target: Target values, same length as smiles_list.
        top_n: Number of top scaffolds to report (by frequency).

    Returns:
        DataFrame with columns: scaffold, count, target_mean, target_std,
        target_min, target_max.  Only includes the top_n most frequent
        scaffolds (smaller ones are too noisy for statistics).
    """
    target = np.asarray(target)
    scaffolds = get_scaffolds(smiles_list)

    # Group target values by scaffold
    scaffold_targets: dict[str, list[float]] = {}
    for scaf, y in zip(scaffolds, target):
        if scaf is None:
            continue
        scaffold_targets.setdefault(scaf, []).append(y)

    # Sort by count, take top_n
    sorted_scaffolds = sorted(
        scaffold_targets.items(), key=lambda x: len(x[1]), reverse=True
    )[:top_n]

    rows = []
    for scaf, values in sorted_scaffolds:
        vals = np.array(values)
        rows.append({
            "scaffold": scaf,
            "count": len(vals),
            "target_mean": vals.mean(),
            "target_std": vals.std(),
            "target_min": vals.min(),
            "target_max": vals.max(),
        })

    result = pd.DataFrame(rows)
    if len(result) > 1:
        mean_range = result["target_mean"].max() - result["target_mean"].min()
        logger.info(
            f"Per-scaffold target stats (top {top_n}): "
            f"mean range = {mean_range:.3f}"
        )
    return result


# ---------------------------------------------------------------------------
# Full scaffold analysis report
# ---------------------------------------------------------------------------

def scaffold_analysis(
    smiles_a: list[str],
    smiles_b: list[str] | None = None,
    target_a: np.ndarray | pd.Series | None = None,
    label_a: str = "train",
    label_b: str = "test",
) -> dict:
    """Run full scaffold analysis and return a W&B-loggable report.

    Args:
        smiles_a: SMILES for split A.
        smiles_b: SMILES for split B (optional — skips overlap if None).
        target_a: Target values for split A (optional — skips per-scaffold stats).
        label_a: Label for split A.
        label_b: Label for split B.

    Returns:
        Dict with keys:
        - ``"diversity_a"``: scaffold_diversity result for split A
        - ``"diversity_b"``: scaffold_diversity result for split B (if given)
        - ``"overlap"``: scaffold_overlap result (if smiles_b given)
        - ``"frequency_table"``: top-20 scaffold frequency table
        - ``"per_scaffold_targets"``: per-scaffold target stats (if target given)
        - Scalar summaries for direct W&B logging
    """
    logger.info(f"Running scaffold analysis ({len(smiles_a)} molecules in {label_a})")

    diversity_a = scaffold_diversity(smiles_a)
    freq_table = scaffold_frequency_table(smiles_a)

    report: dict = {
        "diversity_a": diversity_a,
        "frequency_table": freq_table,
        "scaffold_ratio_a": diversity_a["scaffold_ratio"],
        "singleton_fraction_a": diversity_a["singleton_fraction"],
        "top_10_coverage_a": diversity_a["top_10_coverage"],
    }

    if smiles_b is not None:
        diversity_b = scaffold_diversity(smiles_b)
        overlap = scaffold_overlap(smiles_a, smiles_b, label_a, label_b)
        report["diversity_b"] = diversity_b
        report["overlap"] = overlap
        report["scaffold_ratio_b"] = diversity_b["scaffold_ratio"]
        report["jaccard_overlap"] = overlap["jaccard"]

    if target_a is not None:
        target_stats = scaffold_target_stats(smiles_a, target_a)
        report["per_scaffold_targets"] = target_stats

    return report


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.loaders import load_dataset
    from molgate.data.splits import random_split, scaffold_split

    df = load_dataset("solubility")

    # Compare random vs scaffold split
    random_splits = random_split(df, seed=42)
    scaffold_splits = scaffold_split(df, seed=42)

    print("\n=== Dataset Diversity ===")
    div = scaffold_diversity(df["smiles"].tolist())
    print(f"  Molecules: {div['n_molecules']}")
    print(f"  Unique scaffolds: {div['n_scaffolds']}")
    print(f"  Acyclic: {div['n_acyclic']}")
    print(f"  Scaffold ratio: {div['scaffold_ratio']:.3f}")
    print(f"  Singleton fraction: {div['singleton_fraction']:.1%}")
    print(f"  Top-10 coverage: {div['top_10_coverage']:.1%}")

    print("\n=== Top 10 Scaffolds ===")
    freq = scaffold_frequency_table(df["smiles"].tolist(), top_n=10)
    print(freq.to_string(index=False))

    print("\n=== Random Split — Scaffold Overlap ===")
    rand_overlap = scaffold_overlap(
        random_splits["train"]["smiles"].tolist(),
        random_splits["test"]["smiles"].tolist(),
    )
    print(f"  Shared scaffolds: {rand_overlap['n_shared']}")
    print(f"  Jaccard index: {rand_overlap['jaccard']:.3f}")
    print(f"  Train molecules in shared: {rand_overlap['frac_train_in_shared']:.1%}")

    print("\n=== Scaffold Split — Scaffold Overlap ===")
    scaf_overlap = scaffold_overlap(
        scaffold_splits["train"]["smiles"].tolist(),
        scaffold_splits["test"]["smiles"].tolist(),
    )
    print(f"  Shared scaffolds: {scaf_overlap['n_shared']}")
    print(f"  Jaccard index: {scaf_overlap['jaccard']:.3f}")
    print(f"  Train molecules in shared: {scaf_overlap['frac_train_in_shared']:.1%}")

    print("\n=== Per-Scaffold Target Stats (top 10) ===")
    tstats = scaffold_target_stats(df["smiles"].tolist(), df["y"].values)
    print(tstats.to_string(index=False))

    import IPython; IPython.embed()
