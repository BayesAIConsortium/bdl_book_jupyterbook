#!/usr/bin/env python3
"""Convert Chapter 7 (Laplace introduction) to native MyST Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

from chapter_conversion import ChapterConfig, convert_chapter
from common import tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=tex_path("laplace_approximations", "intro", "main.tex"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("content/laplace_approximations/intro.md"),
    )
    args = parser.parse_args()
    convert_chapter(
        ChapterConfig(
            title="Introduction",
            label="cha:laplace:introduction",
            input_path=args.input,
            output_path=args.output,
            asset_slug="laplace_intro",
        )
    )


if __name__ == "__main__":
    main()
