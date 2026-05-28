"""Activity cliff detection — pairs of structurally similar molecules with divergent activity.

An **activity cliff** is a pair of molecules that are structurally very similar
(high Tanimoto similarity) but have very different biological activity (large
difference in target value).  They are one of the hardest challenges in
molecular property prediction because small structural changes cause large
property changes — exactly the opposite of the smoothness assumption that
most ML models rely on.

Why activity cliffs matter for our project:
  1. **Model evaluation**: Models that correctly predict cliff molecules are
     genuinely learning structure-activity relationships, not just memorising
     neighborhood averages.
  2. **Bias impact**: Biased datasets may systematically remove one side of
     cliff pairs (e.g., dropping extreme target values removes the "high"
     partner).  This makes the remaining data look smoother and inflates
     model performance on the biased set — but the model will fail on the
     full distribution where cliffs exist.
  3. **Dataset characterisation**: The density of activity cliffs tells us
     how "rugged" the structure-activity landscape is.  Solubility tends to
     be smoother than, say, hERG toxicity.

How we detect cliffs:
  For each pair of molecules, compute:
    - **Tanimoto similarity** on Morgan fingerprints (same as in bias.py)
    - **Activity difference** = |y_i - y_j|

  A pair is a cliff if:
    similarity >= sim_threshold  AND  |y_i - y_j| >= act_threshold

  Computing all N*(N-1)/2 pairs is O(N^2), so for large datasets we use
  RDKit's ``BulkTanimotoSimilarity`` for efficient pairwise computation
  (one-vs-many in a single C++ call).

Key metrics:
  - **n_cliff_pairs**: total number of cliff pairs
  - **cliff_fraction**: cliff_pairs / total_similar_pairs (how many similar
    pairs are cliffs vs just similar with similar activity)
  - **cliff_molecules**: unique molecules involved in at least one cliff pair
  - **cliff_molecule_fraction**: cliff_molecules / n_molecules
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CliffPair:
    """A single activity cliff pair."""

    idx_a: int                # Index of molecule A in the input list
    idx_b: int                # Index of molecule B
    smiles_a: str             # SMILES of molecule A
    smiles_b: str             # SMILES of molecule B
    similarity: float         # Tanimoto similarity (high = structurally similar)
    activity_a: float         # Target value for A
    activity_b: float         # Target value for B
    activity_diff: float      # |activity_a - activity_b| (high = divergent)


# ---------------------------------------------------------------------------
# Fingerprint computation (reusable within this module)
# ---------------------------------------------------------------------------

def _compute_morgan_fps(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> list[DataStructs.ExplicitBitVect | None]:
    """Compute Morgan fingerprints, returning None for unparseable SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps


# ---------------------------------------------------------------------------
# Core cliff detection
# ---------------------------------------------------------------------------

def find_cliffs(
    smiles_list: list[str],
    target: np.ndarray | pd.Series,
    sim_threshold: float = 0.7,
    act_threshold: float | None = None,
    act_quantile: float = 0.9,
) -> list[CliffPair]:
    """Find all activity cliff pairs in a dataset.

    A cliff pair has Tanimoto similarity >= sim_threshold AND activity
    difference >= act_threshold.

    Args:
        smiles_list: List of canonical SMILES.
        target: Target values, same length as smiles_list.
        sim_threshold: Minimum Tanimoto similarity to consider a pair
            structurally similar (default 0.7 = quite similar).
        act_threshold: Minimum |activity_diff| to call a pair a cliff.
            If None, computed automatically as the act_quantile of all
            pairwise activity differences among similar pairs.
        act_quantile: Quantile of pairwise activity differences (among
            similar pairs) to use as act_threshold when act_threshold
            is None.  Default 0.9 = top 10% of differences are cliffs.

    Returns:
        List of CliffPair objects, sorted by activity_diff descending
        (most dramatic cliffs first).
    """
    target = np.asarray(target, dtype=float)
    n = len(smiles_list)
    if n != len(target):
        raise ValueError(f"Length mismatch: {n} SMILES vs {len(target)} targets")

    logger.info(
        f"Searching for activity cliffs in {n} molecules "
        f"(sim >= {sim_threshold})"
    )

    # Compute fingerprints
    fps = _compute_morgan_fps(smiles_list)

    # Find all similar pairs and their activity differences
    similar_pairs: list[tuple[int, int, float, float]] = []  # (i, j, sim, diff)

    for i in range(n):
        if fps[i] is None:
            continue
        # BulkTanimotoSimilarity computes similarity of fps[i] against
        # fps[i+1:] in a single C++ call — much faster than pairwise Python
        remaining_fps = [fp for fp in fps[i + 1:] if fp is not None]
        remaining_indices = [j for j in range(i + 1, n) if fps[j] is not None]

        if not remaining_fps:
            continue

        sims = DataStructs.BulkTanimotoSimilarity(fps[i], remaining_fps)

        for sim, j in zip(sims, remaining_indices):
            if sim >= sim_threshold:
                diff = abs(target[i] - target[j])
                similar_pairs.append((i, j, sim, diff))

    logger.info(f"Found {len(similar_pairs)} similar pairs (sim >= {sim_threshold})")

    if not similar_pairs:
        logger.info("No similar pairs found — try lowering sim_threshold")
        return []

    # Determine activity threshold
    if act_threshold is None:
        diffs = np.array([d for _, _, _, d in similar_pairs])
        act_threshold = float(np.quantile(diffs, act_quantile))
        logger.info(
            f"Auto activity threshold: {act_threshold:.3f} "
            f"({act_quantile:.0%} quantile of similar-pair differences)"
        )

    # Filter to cliff pairs
    cliffs = []
    for i, j, sim, diff in similar_pairs:
        if diff >= act_threshold:
            cliffs.append(CliffPair(
                idx_a=i,
                idx_b=j,
                smiles_a=smiles_list[i],
                smiles_b=smiles_list[j],
                similarity=sim,
                activity_a=float(target[i]),
                activity_b=float(target[j]),
                activity_diff=diff,
            ))

    # Sort by activity difference descending
    cliffs.sort(key=lambda c: c.activity_diff, reverse=True)

    logger.info(
        f"Activity cliffs: {len(cliffs)} pairs "
        f"(act_diff >= {act_threshold:.3f}, sim >= {sim_threshold})"
    )
    return cliffs


# ---------------------------------------------------------------------------
# Cliff summary statistics
# ---------------------------------------------------------------------------

def cliff_summary(
    cliffs: list[CliffPair],
    n_molecules: int,
) -> dict:
    """Compute summary statistics from detected cliffs.

    Args:
        cliffs: List of CliffPair objects (from find_cliffs).
        n_molecules: Total number of molecules in the dataset.

    Returns:
        Dict with keys:
        - ``n_cliff_pairs``: total cliff pair count
        - ``n_cliff_molecules``: unique molecules in at least one cliff
        - ``cliff_molecule_fraction``: cliff_molecules / n_molecules
        - ``mean_cliff_sim``: average similarity of cliff pairs
        - ``mean_cliff_diff``: average activity difference of cliff pairs
        - ``max_cliff_diff``: largest activity difference
        - ``most_dramatic``: the single most dramatic cliff (CliffPair)
    """
    if not cliffs:
        return {
            "n_cliff_pairs": 0,
            "n_cliff_molecules": 0,
            "cliff_molecule_fraction": 0.0,
            "mean_cliff_sim": 0.0,
            "mean_cliff_diff": 0.0,
            "max_cliff_diff": 0.0,
            "most_dramatic": None,
        }

    # Unique molecules involved in cliffs
    cliff_mol_indices = set()
    for c in cliffs:
        cliff_mol_indices.add(c.idx_a)
        cliff_mol_indices.add(c.idx_b)

    sims = np.array([c.similarity for c in cliffs])
    diffs = np.array([c.activity_diff for c in cliffs])

    return {
        "n_cliff_pairs": len(cliffs),
        "n_cliff_molecules": len(cliff_mol_indices),
        "cliff_molecule_fraction": len(cliff_mol_indices) / n_molecules,
        "mean_cliff_sim": float(sims.mean()),
        "mean_cliff_diff": float(diffs.mean()),
        "max_cliff_diff": float(diffs.max()),
        "most_dramatic": cliffs[0],  # Already sorted by diff descending
    }


# ---------------------------------------------------------------------------
# Cliff analysis for a dataset
# ---------------------------------------------------------------------------

def cliff_analysis(
    smiles_list: list[str],
    target: np.ndarray | pd.Series,
    sim_threshold: float = 0.7,
    act_threshold: float | None = None,
    act_quantile: float = 0.9,
) -> dict:
    """Run full cliff analysis and return a W&B-loggable report.

    Args:
        smiles_list: List of canonical SMILES.
        target: Target values.
        sim_threshold: Tanimoto similarity threshold.
        act_threshold: Activity difference threshold (None = auto).
        act_quantile: Quantile for auto threshold.

    Returns:
        Dict with:
        - ``"cliffs"``: list of CliffPair objects
        - ``"summary"``: cliff_summary dict
        - Scalar keys for direct W&B logging
    """
    cliffs = find_cliffs(
        smiles_list, target,
        sim_threshold=sim_threshold,
        act_threshold=act_threshold,
        act_quantile=act_quantile,
    )

    summary = cliff_summary(cliffs, n_molecules=len(smiles_list))

    report = {
        "cliffs": cliffs,
        "summary": summary,
        "n_cliff_pairs": summary["n_cliff_pairs"],
        "n_cliff_molecules": summary["n_cliff_molecules"],
        "cliff_molecule_fraction": summary["cliff_molecule_fraction"],
    }

    return report


# ---------------------------------------------------------------------------
# Interactive demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from molgate.data.loaders import load_dataset

    df = load_dataset("solubility")
    smiles = df["smiles"].tolist()
    target = df["y"].values

    # Run cliff detection
    report = cliff_analysis(smiles, target, sim_threshold=0.7)
    summary = report["summary"]

    print("\n=== Activity Cliff Summary ===")
    print(f"  Total cliff pairs: {summary['n_cliff_pairs']}")
    print(f"  Cliff molecules: {summary['n_cliff_molecules']} / {len(smiles)}")
    print(f"  Cliff molecule fraction: {summary['cliff_molecule_fraction']:.1%}")
    print(f"  Mean cliff similarity: {summary['mean_cliff_sim']:.3f}")
    print(f"  Mean activity difference: {summary['mean_cliff_diff']:.3f}")
    print(f"  Max activity difference: {summary['max_cliff_diff']:.3f}")

    if summary["most_dramatic"] is not None:
        mc = summary["most_dramatic"]
        print(f"\n=== Most Dramatic Cliff ===")
        print(f"  Molecule A: {mc.smiles_a}")
        print(f"  Molecule B: {mc.smiles_b}")
        print(f"  Similarity: {mc.similarity:.3f}")
        print(f"  Activity A: {mc.activity_a:.3f}")
        print(f"  Activity B: {mc.activity_b:.3f}")
        print(f"  Difference: {mc.activity_diff:.3f}")

    # Show top 5 cliffs
    if report["cliffs"]:
        print(f"\n=== Top 5 Cliffs ===")
        for i, c in enumerate(report["cliffs"][:5]):
            print(
                f"  {i+1}. sim={c.similarity:.3f}  "
                f"diff={c.activity_diff:.3f}  "
                f"[{c.activity_a:.2f} vs {c.activity_b:.2f}]"
            )

    import IPython; IPython.embed()
