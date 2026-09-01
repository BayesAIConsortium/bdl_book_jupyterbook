#!/usr/bin/env bash
set -euo pipefail

uv run python scripts/convert_preface.py
uv run python scripts/convert_readers_guide.py
uv run python scripts/fix_readers_guide_blocks.py
uv run python scripts/convert_authors.py
uv run python scripts/convert_leadership_and_ai.py
uv run python scripts/convert_acknowledgements.py
