#!/usr/bin/env python3
"""Convert the LaTeX contributor list to a simple MyST/Markdown author page.

By default this reads from the configured TeX source repository and writes
``content/frontmatter/authors.md``. Both paths can be overridden on the
command line.
"""

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


def parse_braced(text: str, start: int) -> tuple[str, int]:
    """Return the contents of a balanced {...} group and the next position."""
    if start >= len(text) or text[start] != "{":
        raise ValueError(f"Expected '{{' at position {start}")

    depth = 0
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif char == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos], pos + 1

    raise ValueError("Unclosed brace group")


def extract_contributors(text: str) -> list[tuple[str, str, str]]:
    contributors: list[tuple[str, str, str]] = []
    marker = "\\contributor"
    pos = 0

    while True:
        pos = text.find(marker, pos)
        if pos < 0:
            break
        pos += len(marker)

        args: list[str] = []
        for _ in range(3):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            value, pos = parse_braced(text, pos)
            args.append(value)

        contributors.append((args[0], args[1], args[2]))

    return contributors


def latex_to_text(value: str) -> str:
    """Convert the small subset of LaTeX used in contributor metadata."""
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

    value = re.sub(r"\\textit\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", value)
    value = value.replace("\\,", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def render_markdown(contributors: list[tuple[str, str, str]]) -> str:
    lines = ["# BayesAI Consortium authors", ""]
    for name, institution, location in contributors:
        lines.extend(
            [
                f"**{latex_to_text(name)}**  ",
                f"{latex_to_text(institution)}  ",
                latex_to_text(location),
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=tex_path("frontmatter", "authors.tex"),
        help="Path to the source authors.tex file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("content/frontmatter/authors.md"),
        help="Path to the generated Markdown file.",
    )
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    contributors = extract_contributors(source)
    if not contributors:
        raise SystemExit(f"No \\contributor entries found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(contributors), encoding="utf-8")
    print(f"Wrote {len(contributors)} contributors to {args.output}")


if __name__ == "__main__":
    main()
