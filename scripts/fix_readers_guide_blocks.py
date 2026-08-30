#!/usr/bin/env python3
"""Normalize generated Reader's Guide table blocks."""

from pathlib import Path
import re


PATH = Path("content/frontmatter/readers_guide.md")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    # Keep opening table directives intact and on their own block boundary.
    # A previous normalization accidentally changed `:::{table}` into
    # `:::\n\n{table}`, so MyST no longer recognized the directive.
    text = re.sub(r"\n*:::\s*\n+\{table\}", "\n\n:::{table}", text)
    text = re.sub(r"\s*:::\{table\}", "\n\n:::{table}", text)

    # The table directive's native :label: metadata is sufficient for MyST
    # cross-references. Remove any standalone targets left by older generated
    # output so the label is defined exactly once.
    text = re.sub(
        r"\n*\((tab:readerguide:[^)]+)\)=\n(?=:::\{table\}[^\n]*\n:label:\s*\1\n)",
        "\n\n",
        text,
    )

    # The tabularx column specification contains nested braces; the lightweight
    # converter currently lets part of that specification leak into the first
    # header cell. Remove it from the generated Markdown header.
    text = re.sub(
        r"^\| c >.*? Ch\. \| Topic \| Read first \| Also useful \|$",
        "| Ch. | Topic | Read first | Also useful |",
        text,
        flags=re.MULTILINE,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    PATH.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"Normalized table directives in {PATH}")


if __name__ == "__main__":
    main()
