#!/usr/bin/env python3
"""Convert Chapter 6 (Discrete sampling) from LaTeX to native MyST Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

from chapter_conversion import ChapterConfig, convert_chapter
from common import tex_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=tex_path("sampling_methods", "discrete_sampling", "main.tex"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("content/sampling_methods/discrete_sampling.md"),
        help="Chapter 6 output used by the normal book TOC.",
    )
    args = parser.parse_args()
    convert_chapter(
        ChapterConfig(
            title="Discrete sampling",
            label="chap:sampling_methods_discrete_sampling",
            input_path=args.input,
            output_path=args.output,
            asset_slug="discrete_sampling",
        )
    )


if __name__ == "__main__":
    main()
