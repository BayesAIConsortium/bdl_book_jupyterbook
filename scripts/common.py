#!/usr/bin/env python3
"""Shared paths and helpers for TeX-to-MyST conversion scripts."""

from __future__ import annotations

import os
from pathlib import Path


TEX_ROOT = Path(os.environ.get("BDL_TEX_ROOT", "../bdl_book_tex_fork"))

# Editorial display titles, keyed by stable chapter labels. Keeping exact
# spellings here preserves acronyms and proper names during every regeneration.
CHAPTER_TITLE_OVERRIDES = {
    "chap:sampling_methods_sg_mcmc": "Stochastic gradient MCMC",
}


def display_chapter_title(label: str, source_title: str) -> str:
    """Return a chapter's editorial title, falling back to its source title."""
    return CHAPTER_TITLE_OVERRIDES.get(label, source_title)


def tex_path(*parts: str) -> Path:
    """Return a path inside the configured TeX source repository."""
    return TEX_ROOT.joinpath(*parts)
