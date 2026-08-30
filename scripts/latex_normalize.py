#!/usr/bin/env python3
"""Shared normalization for book-specific LaTeX before MyST conversion.

This module handles reusable notation expansion and non-semantic typographic
cleanup. Semantic structures such as algorithms, theorems, proofs, and chapter
headings belong in myst_structures.py.
"""

from __future__ import annotations

import re


COMPOUND_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\network(\inputs,\params)", r"f(x,\theta)"),
)

SIMPLE_MATH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\traininputs", "X"),
    (r"\traintargets", "Y"),
    (r"\traindata", r"\mathcal{D}"),
    (r"\testinput", r"x_*"),
    (r"\testoutput", r"y_*"),
    (r"\inputspace", r"\mathcal{X}"),
    (r"\targetspace", r"\mathcal{Y}"),
    (r"\paramspace", r"\Theta"),
    (r"\diminputs", "D"),
    (r"\dimoutputs", "C"),
    (r"\dimparams", "P"),
    (r"\numtraindata", "N"),
    (r"\numMCsamples", "M"),
    (r"\probability", "p"),
    (r"\approxprob", "q"),
    (r"\targetprob", r"\pi"),
    (r"\testfunction", "h"),
    (r"\unnormalized", r"\kappa"),
    (r"\mcmckernelB", r"\mathcal{K}"),
    (r"\mcmckernel", r"\mathcal{M}"),
    (r"\partitionfunc", "Z"),
    (r"\network", "f"),
    (r"\activation", r"\sigma"),
    (r"\softmax", "S"),
    (r"\netdepth", "L"),
    (r"\loss", r"\ell"),
    (r"\risk", r"\mathcal{L}"),
    (r"\regularizer", r"\Omega"),
    (r"\Reals", r"\mathbb{R}"),
    (r"\Nats", r"\mathbb{N}"),
    (r"\params", r"\theta"),
    (r"\inputs", "x"),
    (r"\targets", "y"),
    (r"\weights", "A"),
    (r"\biases", "b"),
    (r"\constant", "K"),
)


TYPOGRAPHIC_COMMANDS: tuple[str, ...] = (
    r"\smallskip",
    r"\medskip",
    r"\bigskip",
)


def normalize_notation(text: str) -> str:
    """Expand the conservative subset of global book notation macros."""
    for source, replacement in COMPOUND_REPLACEMENTS:
        text = text.replace(source, replacement)
    for source, replacement in SIMPLE_MATH_REPLACEMENTS:
        text = text.replace(source, replacement)
    return text


def strip_tex_comments(text: str) -> str:
    """Remove ordinary TeX comments while retaining escaped percent signs."""
    return re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)


def normalize_typography(text: str) -> str:
    """Drop non-semantic TeX presentation commands while retaining their text."""
    text = re.sub(
        r"\\chapterinitial\{([^{}]+)\}\{([^{}]+)\}",
        lambda match: match.group(1) + match.group(2),
        text,
    )
    for command in TYPOGRAPHIC_COMMANDS:
        text = text.replace(command, "")
    return text


def normalize_latex(text: str) -> str:
    """Normalize comments, notation, and non-semantic typography before MyST conversion."""
    text = strip_tex_comments(text)
    text = normalize_notation(text)
    text = normalize_typography(text)
    return text
