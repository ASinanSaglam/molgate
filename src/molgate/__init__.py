"""molgate — Molecular property prediction with dataset bias experiments."""

import warnings

# Suppress torch-sparse CUDA warning from PyG — we use edge_index format,
# not SparseTensor, so torch-sparse is not needed.
warnings.filterwarnings("ignore", message=".*torch-sparse.*")

# Single-source version: matches pyproject.toml's `version = "0.1.0"`.
# Consumers can do `import molgate; print(molgate.__version__)` to check
# which version they have installed.  If we later want truly single-source
# versioning (auto-read from pyproject.toml at build time), hatchling
# supports dynamic versioning — but a hardcoded string is simpler to start.
__version__ = "0.1.0"
