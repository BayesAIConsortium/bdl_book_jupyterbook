# Handbook of Bayesian Deep Learning

This repository hosts the Jupyter Book 2 / MyST edition of the *Handbook of Bayesian Deep Learning*.

## Local setup with uv

Install the project environment, including development tools:

```bash
uv sync --extra dev
```

Verify the main tools:

```bash
uv run jupyter-book --version
uv run myst --version
```

## Pull changes and preview locally

Update the local repository, synchronize the environment, regenerate the migrated Markdown content, and start the local preview server:

```bash
git pull
uv sync --extra dev
bash scripts/run_conversions.sh
uv run jupyter book start
```

When dependencies in `pyproject.toml` change, run `uv sync --extra dev` again.

The repository is intentionally book-first rather than a reusable Python package. Any TeX-to-MyST migration helpers we add initially will live under `scripts/`.
