#!/usr/bin/env python3
"""Convert the editorial leadership and AI disclosure front matter to MyST."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import tex_path


WEB_VERSION_DISCLOSURE = (
    "**Web version of the book.** Theodore Papamarkou converted the book from its TeX "
    "source to this Jupyter Book web edition with the assistance of GPT."
)

RESPONSIBILITY_STATEMENT = (
    "Responsibility for the content of this book, including any errors that remain, "
    "rests with its human authors and editors."
)


def clean_inline(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    text = text.replace("---", "—")
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"**\1**", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def convert(text: str) -> str:
    text = re.sub(
        r"^\\chapter\*\{Editorial leadership and AI disclosure\}\s*",
        "",
        text,
        count=1,
    )

    parts = re.split(r"\\section\*\{([^{}]+)\}", text)
    if len(parts) < 3:
        raise ValueError("Could not find expected sections")

    output = ["# Editorial leadership and AI disclosure"]
    final_paragraph: str | None = None

    for i in range(1, len(parts), 2):
        title = clean_inline(parts[i])
        body = parts[i + 1]
        body = re.sub(
            r"\\paragraph\{([^{}]+)\}",
            lambda m: f"\n\n**{clean_inline(m.group(1))}** ",
            body,
        )
        paragraphs = [clean_inline(p) for p in re.split(r"\n\s*\n", body) if clean_inline(p)]

        if title == "Use of AI tools" and paragraphs:
            if paragraphs[-1] != RESPONSIBILITY_STATEMENT:
                raise ValueError("Expected responsibility statement at end of AI disclosure")
            final_paragraph = paragraphs.pop()

        output.append(f"## {title}")
        output.extend(paragraphs)

    if final_paragraph is None:
        raise ValueError("Could not find responsibility statement in AI disclosure")

    output.extend([WEB_VERSION_DISCLOSURE, final_paragraph])
    return "\n\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=tex_path("frontmatter", "leadership_and_ai.tex"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("content/frontmatter/leadership_and_ai.md"),
    )
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(convert(source), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
