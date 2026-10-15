#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/convert_sampling_intro.py
uv run python scripts/convert_sg_mcmc.py
uv run python scripts/convert_sbi.py
uv run python scripts/convert_multilevel_smc.py
uv run python scripts/convert_low_precision_sampling.py
uv run python scripts/convert_discrete_sampling.py
