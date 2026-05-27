# Makefile — convenience targets for molgate development.
#
# Usage:
#   make install      Install package in editable mode with dev dependencies
#   make test         Run the test suite
#   make lint         Check code style and type errors (no auto-fix)
#   make format       Auto-fix code style issues
#   make notebooks    Execute all notebooks in place (for CI or refresh)

.PHONY: install test lint format notebooks clean

# Install the package in editable mode so `import molgate` resolves to src/.
# The [dev] extra pulls in pytest, ruff, mypy, etc.
install:
	pip install -e ".[dev]"

# Run all tests. pytest finds tests/ automatically via pyproject.toml config.
test:
	pytest

# Lint: ruff checks style rules, mypy checks types.
# Neither modifies files — safe to run in CI.
lint:
	ruff check src/ tests/
	mypy src/

# Format: ruff auto-fixes style issues and sorts imports.
format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# Execute all notebooks top-to-bottom and overwrite outputs in place.
# Useful before committing to ensure notebooks are up to date.
notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# Remove build artifacts and caches.
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
