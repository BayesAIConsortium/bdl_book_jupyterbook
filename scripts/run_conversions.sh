#!/usr/bin/env bash
set -euo pipefail

git pull

uv run python scripts/generate_book_skeleton.py
bash scripts/run_frontmatter_conversions.sh
bash scripts/run_part1_conversions.sh
