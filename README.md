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

## MyST project root and URLs

The MyST project root is `content/`, with its configuration in `content/myst.yml`. Keeping the project root there means the public URL hierarchy mirrors the book structure without an extra `/content/` prefix. With `folders: true`, for example:

```text
content/sampling_methods/index.md  -> /sampling-methods/
content/sampling_methods/intro.md  -> /sampling-methods/intro/
content/laplace_approximations/intro.md -> /laplace-approximations/intro/
```

The central bibliography remains at the repository root as `references.bib` and is referenced from `content/myst.yml` as `../references.bib`.

## Conversion workflow

The TeX repository remains authoritative during the migration. By default, the conversion scripts read it from `../bdl_book_tex_fork`. Override that location with `BDL_TEX_ROOT` when needed.

Run the complete set of currently integrated conversions with:

```bash
bash scripts/run_conversions.sh
```

The skeleton generator maintains linked chapter lists on all ten part overview pages. Chapter numbers follow the order in the LaTeX book, continuing across parts. Lists are refreshed on each run while authored overview text is preserved; chapters not yet migrated link to their existing placeholder pages.

The top-level runner first refreshes the generated book skeleton, then delegates to smaller book-wide groups:

```bash
bash scripts/run_frontmatter_conversions.sh
bash scripts/run_part1_conversions.sh
bash scripts/run_part2_conversions.sh
```

`run_part2_conversions.sh` currently converts Chapters 7–11: the introduction to Laplace approximations, linearised Laplace approximations, the differential geometric perspective, analytic functionality, and model selection with the marginal likelihood. To regenerate one alone, run `uv run python scripts/convert_laplace_intro.py`, `uv run python scripts/convert_laplace_linearized.py`, `uv run python scripts/convert_laplace_diffgeo.py`, `uv run python scripts/convert_laplace_analytic_functionality.py`, or `uv run python scripts/convert_laplace_differentiable.py`. Native MyST subfigures preserve panel labels and use image directives at 45% width, consistent with Part 1; simple booktabs tables retain their header row. Callouts become native MyST admonitions; aligned subequations become separate auto-numbered math directives preserving each row label. Project math macros live in `content/myst.yml`. Chapter 11 explicitly sets a 45% fallback for its two-image figure because the TeX supplies no widths; other chapter converters retain their defaults.

`run_frontmatter_conversions.sh` converts the migrated front matter. `run_part1_conversions.sh` converts all six sampling chapters, from *Introduction to sampling* through *Discrete sampling*. As more chapters are migrated, their converters should be added to the corresponding part runner; new part runners can then be called from `run_conversions.sh`. Shared TeX-to-MyST behavior belongs in reusable modules such as `latex_normalize.py` and `myst_structures.py`, rather than in chapter-specific patches whenever possible.

To regenerate Chapter 6 alone, run `uv run python scripts/convert_discrete_sampling.py` from the repository root. Figure copying automatically prefers matching SVG, PNG, or other web images over the source PDF. Section references use MyST's native `{name}` substitution, so links to headings in other chapters display their current titles automatically.

Chapter converters read the full title from the source `\chapter[short title]{Full title}` command on every run. The optional short title does not replace the page title. Editorial chapter titles can be set in `CHAPTER_TITLE_OVERRIDES` in `scripts/common.py`, keyed by the chapter's stable label. Both the skeleton generator and chapter converters use these exact titles, keeping chapter lists, headings, and sidebar entries consistent across regeneration while preserving acronyms and proper names.

## Pull changes and preview locally

For the normal local preview workflow, run from the repository root:

```bash
git pull
bash scripts/run_conversions.sh
cd content
uv run jupyter book start
```

Run `uv sync --extra dev` first when setting up a fresh checkout or whenever dependencies in `pyproject.toml` change.

For a production HTML build, run from the repository root:

```bash
cd content
uv run jupyter book build --html
```

## Netlify

Because the MyST project root is `content/`, Netlify must build from that directory while still installing the Python project from the repository root. Use:

```text
Build command: python -m pip install . && cd content && jupyter book build --html
Publish directory: content/_build/html
```

The Netlify base directory should remain blank / at the repository root.

The repository is intentionally book-first rather than a reusable Python package. TeX-to-MyST migration helpers live under `scripts/`.
