"""molgate — Molecular property prediction with dataset bias experiments."""

import warnings

# Suppress all optional PyG CUDA extension warnings.
# These .so files are compiled against a different torch ABI version, but PyG
# gracefully falls back to native PyTorch ops. We only use standard edge_index
# format + GINEConv, which don't require any of these optional extensions.
warnings.filterwarnings("ignore", message=".*torch-sparse.*")
warnings.filterwarnings("ignore", message=".*torch-spline-conv.*")
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
warnings.filterwarnings("ignore", message=".*torch-cluster.*")

# Single-source version: matches pyproject.toml's `version = "0.1.0"`.
# Consumers can do `import molgate; print(molgate.__version__)` to check
# which version they have installed.  If we later want truly single-source
# versioning (auto-read from pyproject.toml at build time), hatchling
# supports dynamic versioning — but a hardcoded string is simpler to start.
__version__ = "0.1.0"
