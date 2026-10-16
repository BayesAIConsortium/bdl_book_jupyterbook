#!/usr/bin/env python3
"""Shared normalization for TeX text encodings used across the book."""

from __future__ import annotations

import re


TEX_TEXT_ACCENTS: dict[str, dict[str, str]] = {
    "'": {
        "a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "y": "ý",
        "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú", "Y": "Ý",
    },
    '"': {
        "a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü", "y": "ÿ",
        "A": "Ä", "E": "Ë", "I": "Ï", "O": "Ö", "U": "Ü",
    },
    "`": {
        "a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù",
        "A": "À", "E": "È", "I": "Ì", "O": "Ò", "U": "Ù",
    },
    "^": {
        "a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û",
        "A": "Â", "E": "Ê", "I": "Î", "O": "Ô", "U": "Û",
    },
    "~": {
        "a": "ã", "n": "ñ", "o": "õ", "A": "Ã", "N": "Ñ", "O": "Õ",
    },
    "c": {"c": "ç", "C": "Ç"},
}

_ACCENT_PATTERN = re.compile(r"\\([\'\"`\^~c])(?:\{([^{}])\}|([^\\{}\s]))")


def normalize_tex_text_accents(text: str) -> str:
    r"""Convert supported TeX text accents to Unicode characters.

    Both braced and unbraced text forms are accepted, for example ``\"{u}``
    and ``\"u``. Mathematical accent commands such as ``\tilde{x}``,
    ``\hat{x}``, and ``\bar{x}`` are distinct commands and are left untouched.
    Unknown accent/character combinations are preserved verbatim.
    """

    def replace_accent(match: re.Match[str]) -> str:
        accent = match.group(1)
        letter = match.group(2) or match.group(3)
        return TEX_TEXT_ACCENTS.get(accent, {}).get(letter, match.group(0))

    return _ACCENT_PATTERN.sub(replace_accent, text)



def normalize_tex_citation_note(text: str) -> str:
    r"""Convert TeX prose spacing and accents in citation pre/postnotes."""
    text = normalize_tex_text_accents(text)
    text = re.sub(r"(?<!\\)~|\\(?:[,;: ]|qquad\b|quad\b|space\b)", " ", text)
    text = text.replace(r"\!", "")
    return re.sub(r"\s+", " ", text).strip()
