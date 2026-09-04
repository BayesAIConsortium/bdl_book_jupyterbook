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

    # Preserve the author's capitalization for the full word "equation", while
    # normalizing the abbreviation "Eq." to the explicit semantic word.
    text = re.sub(
        r"\b([Ee]quation|[Ee]q\.)\s*~?\s*\\(?:eqref|ref)\{([^{}]+)\}",
        lambda m: placeholder(
            f"{'equation' if m.group(1) == 'equation' else 'Equation'} [](#{m.group(2)})"
        ),
        text,
    )
    text = re.sub(
        r"\\eqref\{([^{}]+)\}",
        lambda m: placeholder(f"Equation [](#{m.group(1)})"),
        text,
    )

    # Section references use the target title where available because web sections
    # are not guaranteed to expose stable numeric enumeration.
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

    text = re.sub(
        r"\b(Chapter|Chapters)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"{m.group(1)} [](#{m.group(2)})"),
        text,
    )
    return text, structures
