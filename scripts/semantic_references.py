#!/usr/bin/env python3
"""Reusable semantic conversion of TeX cross-references to MyST links."""

from __future__ import annotations

import re

from myst_structures import ExtractedStructure


def extract_references(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Protect semantic TeX references with explicit MyST-friendly display text."""
    structures: list[ExtractedStructure] = []

    def placeholder(markdown: str) -> str:
        token = f"BDLREFERENCEPLACEHOLDER{len(structures):04d}"
        structures.append(ExtractedStructure(token, markdown))
        return token

    # When the source explicitly supplies the semantic word, preserve it. This
    # covers both \eqref and \ref while letting MyST supply the equation number.
    text = re.sub(
        r"\b([Ee]quation|[Ee]q\.)\s*~?\s*\\(?:eqref|ref)\{([^{}]+)\}",
        lambda m: placeholder(
            f"{'equation' if m.group(1) == 'equation' else 'Equation'} [](#{m.group(2)})"
        ),
        text,
    )

    # A bare \eqref semantically contributes only the parenthesized equation
    # number. Do not invent the word ``Equation`` when it was absent in TeX.
    text = re.sub(
        r"\\eqref\{([^{}]+)\}",
        lambda m: placeholder(f"[](#{m.group(1)})"),
        text,
    )

    # MyST resolves the heading title across the whole book, including targets
    # in other chapters. Avoid freezing a title or inserting a "link" fallback.
    text = re.sub(
        r"\b(?:Section|Sec\.)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(
            f"[Section {{name}}](#{m.group(1)})"
        ),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:Figure|Fig\.)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"[](#{m.group(1)})"),
        text,
        flags=re.IGNORECASE,
    )

    # MyST proof cross-references already render their semantic object prefix
    # (for example, ``Algorithm 2`` or ``Theorem 1``). Preserve the target but
    # do not duplicate that prefix in surrounding Markdown.
    text = re.sub(
        r"\b(?:Algorithm|Theorem|Proposition|Lemma|Definition|Remark|Assumption|Table|Part)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"[](#{m.group(1)})"),
        text,
    )

    text = re.sub(
        r"\b(Chapter|Chapters)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"{m.group(1)} [](#{m.group(2)})"),
        text,
    )

    # Protect remaining references as well: MyST's TeX exporter can otherwise
    # insert literal %s display text or a fixed, incorrect table number.
    text = re.sub(
        r"\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"[](#{m.group(1)})"),
        text,
    )

    # A TeX group used only to scope ordinary prose has no semantic meaning in
    # Markdown. Once its semantic reference is protected by a placeholder, a
    # simple one-line group can be unwrapped safely without touching math groups.
    text = re.sub(
        r"\{([^{}\n]*BDLREFERENCEPLACEHOLDER\d{4}[^{}\n]*)\}",
        r"\1",
        text,
    )

    return text, structures
