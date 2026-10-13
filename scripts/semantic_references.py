#!/usr/bin/env python3
"""Reusable semantic conversion of TeX cross-references to MyST links."""

from __future__ import annotations

import re

from latex_normalize import normalize_notation
from myst_structures import ExtractedStructure


def _clean_inline_tex(text: str) -> str:
    """Normalize conservative inline TeX used in section titles."""
    text = normalize_notation(text)
    text = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = text.replace(r"\&", "&")
    text = text.replace("~", " ")
    text = text.replace("---", "—")
    text = text.replace("--", "–")
    return re.sub(r"\s+", " ", text).strip()


def _section_label_titles(text: str) -> dict[str, str]:
    """Map section-like labels to human-readable titles for web references."""
    titles: dict[str, str] = {}
    pattern = re.compile(
        r"\\(?:section|subsection|subsubsection)\*?\{([^{}]+)\}\s*\\label\{([^{}]+)\}",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        titles[match.group(2)] = _clean_inline_tex(match.group(1))
    return titles


def extract_references(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Protect semantic TeX references with explicit MyST-friendly display text."""
    structures: list[ExtractedStructure] = []
    section_titles = _section_label_titles(text)

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

    text = re.sub(
        r"\b(?:Section|Sec\.)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(
            f"Section [{section_titles.get(m.group(1), 'link')}](#{m.group(1)})"
        ),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:Figure|Fig\.)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"Figure [](#{m.group(1)})"),
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
