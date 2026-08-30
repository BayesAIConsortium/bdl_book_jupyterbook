#!/usr/bin/env python3
"""Convert the LaTeX Preface to the current MyST/Markdown form."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import tex_path


def convert(text: str) -> str:
    text = re.sub(r"^\\chapter\*\{Preface\}\s*", "# Preface\n\n", text, count=1)
    text = text.replace("---", "—")
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=tex_path("frontmatter", "preface.tex"))
    parser.add_argument("--output", type=Path, default=Path("content/frontmatter/preface.md"))
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(convert(source), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
