#!/usr/bin/env python3
"""Convert the LaTeX acknowledgements front matter to MyST/Markdown."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import tex_path


ACCENTS = {
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


def latex_to_text(value: str) -> str:
    value = value.replace("---", "—")
    value = value.replace("--", "–")
    value = value.replace("``", '“').replace("''", '”')
    value = value.replace("\\&", "&")
    value = value.replace("~", " ")

    accent_pattern = re.compile(r"\\([\'\"`\^~c])(?:\{([^{}])\}|([^\\{}\s]))")

    def replace_accent(match: re.Match[str]) -> str:
        accent = match.group(1)
        letter = match.group(2) or match.group(3)
        return ACCENTS.get(accent, {}).get(letter, letter)

    previous = None
    while previous != value:
        previous = value
        value = accent_pattern.sub(replace_accent, value)

    value = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", value)
    value = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def convert(text: str) -> str:
    text = re.sub(r"^\\chapter\*\{Acknowledgements\}\s*", "", text, count=1)
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    paragraphs = [latex_to_text(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "# Acknowledgements\n\n" + "\n\n".join(paragraphs) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=tex_path("frontmatter", "acknowledgements.tex"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("content/frontmatter/acknowledgements.md"),
    )
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(convert(source), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
