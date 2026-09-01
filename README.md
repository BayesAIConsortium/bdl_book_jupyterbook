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

No separate Node/npm setup is required for the proof, theorem, definition, example, proposition, or algorithm directives used by the migrated book. The current Jupyter Book / MyST stack already provides the proof extension and its `prf:*` directives. In particular, do not run `npm install` for this repository solely to enable proofs, and do not commit `node_modules/` or an accidental `package-lock.json` created while experimenting with that approach.

## Conversion workflow

The TeX repository remains authoritative during the migration. By default, the conversion scripts read it from `../bdl_book_tex_fork`. Override that location with `BDL_TEX_ROOT` when needed.

Run the complete set of currently integrated conversions with:

```bash
bash scripts/run_conversions.sh
```

The top-level runner first refreshes the generated book skeleton, then delegates to smaller book-wide groups:

```bash
bash scripts/run_frontmatter_conversions.sh
bash scripts/run_part1_conversions.sh
```

`run_frontmatter_conversions.sh` converts the migrated front matter. `run_part1_conversions.sh` currently converts Chapter 1, *Introduction to sampling*. As more chapters are migrated, their converters should be added to the corresponding part runner; new part runners can then be called from `run_conversions.sh`. Shared TeX-to-MyST behavior belongs in reusable modules such as `latex_normalize.py` and `myst_structures.py`, rather than in chapter-specific patches whenever possible.

## Pull changes and preview locally

Update the local repository, synchronize the environment, regenerate the migrated Markdown content, and start the local preview server:

```bash
git pull
uv sync --extra dev
bash scripts/run_conversions.sh
uv run jupyter book start
```

When dependencies in `pyproject.toml` change, run `uv sync --extra dev` again.

The repository is intentionally book-first rather than a reusable Python package. TeX-to-MyST migration helpers live under `scripts/`.
