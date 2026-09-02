#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/convert_sampling_intro.py
uv run python scripts/convert_sg_mcmc.py
