#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/convert_laplace_intro.py
uv run python scripts/convert_laplace_linearized.py
