"""Command-line interface for molgate.

Usage
-----
    # Single SMILES prediction
    molgate predict solubility "CCO"
    molgate predict pxr "CCO"
    molgate predict all "CCO"           # all properties with checkpoints

    # Batch from CSV
    molgate predict solubility --file molecules.csv
    molgate predict all --file molecules.csv --output predictions.csv

    # List available checkpoints
    molgate models list
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

# Checkpoint root is co-located with the repository, not CWD.
# cli.py lives at src/molgate/cli.py  →  three levels up is the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT_ROOT = _PROJECT_ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="molgate",
        description="Molecular property prediction — molgate CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ predict
    p_pred = sub.add_parser("predict", help="Predict molecular properties")
    p_pred.add_argument(
        "property",
        metavar="PROPERTY",
        help="Property to predict (e.g. solubility, pxr) or 'all'",
    )
    p_pred.add_argument(
        "smiles", nargs="?", default=None,
        metavar="SMILES",
        help="Single SMILES string (omit when using --file)",
    )
    p_pred.add_argument(
        "--file", "-f", default=None, metavar="PATH",
        help="CSV file with a SMILES column",
    )
    p_pred.add_argument(
        "--smiles-col", default="smiles", metavar="COL",
        help="Column name for SMILES in --file (default: smiles)",
    )
    p_pred.add_argument(
        "--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT), metavar="DIR",
        help=f"Root directory for checkpoints (default: {DEFAULT_CHECKPOINT_ROOT})",
    )
    p_pred.add_argument(
        "--output", "-o", default=None, metavar="PATH",
        help="Write predictions CSV here (default: print to stdout)",
    )

    # --------------------------------------------------------------- models list
    p_models = sub.add_parser("models", help="Manage available trained models")
    models_sub = p_models.add_subparsers(dest="models_command", required=True)
    p_list = models_sub.add_parser("list", help="List available checkpoints")
    p_list.add_argument(
        "--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT), metavar="DIR",
        help=f"Root directory to search (default: {DEFAULT_CHECKPOINT_ROOT})",
    )

    args = parser.parse_args()

    if args.command == "predict":
        _cmd_predict(args)
    elif args.command == "models" and args.models_command == "list":
        _cmd_models_list(args)


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _find_best_checkpoint(property_name: str, checkpoint_root: Path) -> Path | None:
    """Return the best checkpoint for a property (lowest test_mae, ties broken by mtime)."""
    candidates = []
    for mf in checkpoint_root.rglob("manifest.json"):
        with open(mf) as f:
            m = json.load(f)
        if m.get("dataset") == property_name:
            mae = m.get("metrics", {}).get("test_mae", float("inf"))
            candidates.append((mae, mf.stat().st_mtime, mf.parent))

    if not candidates:
        return None
    # Sort: lowest MAE first; within same MAE, most recent first
    candidates.sort(key=lambda t: (t[0], -t[1]))
    return candidates[0][2]


def _all_properties(checkpoint_root: Path) -> dict[str, Path]:
    """Return {property_name: best_checkpoint_path} for every property found."""
    properties: dict[str, list[tuple]] = {}
    for mf in checkpoint_root.rglob("manifest.json"):
        with open(mf) as f:
            m = json.load(f)
        dataset = m.get("dataset")
        if not dataset:
            continue
        mae = m.get("metrics", {}).get("test_mae", float("inf"))
        properties.setdefault(dataset, []).append(
            (mae, mf.stat().st_mtime, mf.parent)
        )

    result = {}
    for dataset, candidates in properties.items():
        candidates.sort(key=lambda t: (t[0], -t[1]))
        result[dataset] = candidates[0][2]
    return result


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------

def _cmd_predict(args: argparse.Namespace) -> None:
    import pandas as pd
    from molgate.serve.predict import predict_smiles

    if args.smiles and args.file:
        print("error: provide either a SMILES argument or --file, not both.", file=sys.stderr)
        sys.exit(1)
    if not args.smiles and not args.file:
        print("error: provide a SMILES string or --file.", file=sys.stderr)
        sys.exit(1)

    checkpoint_root = Path(args.checkpoint_root)

    if args.file:
        df_in = pd.read_csv(args.file)
        if args.smiles_col not in df_in.columns:
            print(
                f"error: column {args.smiles_col!r} not found.\n"
                f"       available columns: {list(df_in.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)
        smiles_input: str | list[str] = df_in[args.smiles_col].tolist()
    else:
        smiles_input = args.smiles

    is_all = args.property == "all"

    if is_all:
        ckpts = _all_properties(checkpoint_root)
        if not ckpts:
            print(f"error: no checkpoints found under {checkpoint_root!r}", file=sys.stderr)
            sys.exit(1)
        _predict_all(smiles_input, ckpts, args.output)
    else:
        ckpt = _find_best_checkpoint(args.property, checkpoint_root)
        if ckpt is None:
            print(
                f"error: no checkpoint found for property {args.property!r}.\n"
                f"       available: {sorted(_all_properties(checkpoint_root).keys()) or 'none'}",
                file=sys.stderr,
            )
            sys.exit(1)
        _predict_single_property(smiles_input, ckpt, args.output)


def _print_model_info(ckpt: Path) -> None:
    with open(ckpt / "manifest.json") as f:
        m = json.load(f)
    metrics = m.get("metrics", {})
    m_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    print(f"model: {m.get('model_name','?')}  dataset: {m.get('dataset','?')}  {m_str}",
          file=sys.stderr)


def _predict_single_property(
    smiles_input: str | list[str],
    ckpt: Path,
    output: str | None,
) -> None:
    from molgate.serve.predict import predict_smiles

    _print_model_info(ckpt)
    results = predict_smiles(smiles_input, ckpt)

    if output:
        results.to_csv(output, index=False)
        n_valid = int(results["valid"].sum())
        print(f"{n_valid}/{len(results)} molecules predicted → {output}", file=sys.stderr)
    elif isinstance(smiles_input, str):
        row = results.iloc[0]
        if not row["valid"]:
            print(f"error: could not parse SMILES {row['smiles']!r}", file=sys.stderr)
            sys.exit(1)
        print(f"{row['prediction']:.4f}")
    else:
        print(results.to_csv(index=False), end="")


def _predict_all(
    smiles_input: str | list[str],
    ckpts: dict[str, Path],
    output: str | None,
) -> None:
    import pandas as pd
    from molgate.serve.predict import predict_smiles

    is_single = isinstance(smiles_input, str)
    smiles_list = [smiles_input] if is_single else list(smiles_input)

    combined: pd.DataFrame | None = None

    for prop, ckpt in sorted(ckpts.items()):
        _print_model_info(ckpt)
        results = predict_smiles(smiles_list, ckpt)
        results = results.rename(columns={"prediction": f"prediction_{prop}"})

        if combined is None:
            combined = results[["smiles", "valid", f"prediction_{prop}"]]
        else:
            # valid = True only if parseable for ALL models (should be the same mask)
            combined = combined.merge(
                results[["smiles", f"prediction_{prop}"]],
                on="smiles",
                how="left",
            )

    if combined is None:
        return

    if output:
        combined.to_csv(output, index=False)
        print(f"Predictions for {list(ckpts.keys())} → {output}", file=sys.stderr)
    elif is_single:
        row = combined.iloc[0]
        if not row["valid"]:
            print(f"error: could not parse SMILES {row['smiles']!r}", file=sys.stderr)
            sys.exit(1)
        for prop in sorted(ckpts.keys()):
            print(f"{prop}: {row[f'prediction_{prop}']:.4f}")
    else:
        print(combined.to_csv(index=False), end="")


# ---------------------------------------------------------------------------
# models list
# ---------------------------------------------------------------------------

def _cmd_models_list(args: argparse.Namespace) -> None:
    root = Path(args.checkpoint_root)
    manifests = sorted(
        root.rglob("manifest.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    if not manifests:
        print(f"no checkpoints found under {root!r}")
        return

    header = f"{'name':<30} {'dataset':<15} {'MAE':>7}  path"
    print(header)
    print("-" * len(header))
    for mf in manifests:
        with open(mf) as f:
            m = json.load(f)
        mae = m.get("metrics", {}).get("test_mae", "—")
        mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
        print(
            f"{m.get('model_name','?'):<30} {m.get('dataset','?'):<15} "
            f"{mae_str:>7}  {mf.parent}"
        )
