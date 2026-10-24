#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/convert_laplace_intro.py
uv run python scripts/convert_laplace_linearized.py
uv run python scripts/convert_laplace_diffgeo.py
uv run python scripts/convert_laplace_analytic_functionality.py
uv run python scripts/convert_laplace_differentiable.py
