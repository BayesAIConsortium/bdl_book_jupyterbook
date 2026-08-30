#!/usr/bin/env python3
"""Shared paths and helpers for TeX-to-MyST conversion scripts."""

from __future__ import annotations

import os
from pathlib import Path


TEX_ROOT = Path(os.environ.get("BDL_TEX_ROOT", "../bdl_book_tex_fork"))


def tex_path(*parts: str) -> Path:
    """Return a path inside the configured TeX source repository."""
    return TEX_ROOT.joinpath(*parts)
