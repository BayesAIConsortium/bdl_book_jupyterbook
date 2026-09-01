#!/usr/bin/env python3
"""Convert Chapter 1 (Introduction to sampling) from LaTeX to native MyST Markdown."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from common import TEX_ROOT, tex_path
from latex_normalize import normalize_latex
from myst_structures import (
    extract_algorithms,
    mark_proof_environments,
    normalize_headings_for_latex_pass,
    restore_extracted_structures,
)


CHAPTER_TITLE = "Introduction to sampling"
CHAPTER_LABEL = "chap:sampling:intro"
MYST_TIMEOUT_SECONDS = 60
WEB_IMAGE_EXTENSIONS = (".svg", ".png", ".webp", ".jpg", ".jpeg")
SOURCE_IMAGE_PREFIX = "sampling_methods/intro/fig/"


def prepare_latex(text: str):
    """Apply shared structural extraction and LaTeX normalization."""
    text, structures = extract_algorithms(text)
    text = mark_proof_environments(text)
    text = normalize_latex(text)
    text = normalize_headings_for_latex_pass(text)
    return text, structures


def clean_generated_markdown(text: str, structures) -> str:
    """Restore native MyST structures and apply minimal chapter cleanup."""
    text = re.sub(r"^#\s+Introduction to sampling\s*\n", "", text, count=1)
    text = restore_extracted_structures(text, structures)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return f"({CHAPTER_LABEL})=\n# {CHAPTER_TITLE}\n\n{text.lstrip()}"


def summarize_myst_output(output: str) -> None:
    """Print a compact summary of diagnostics for the isolated normalized file."""
    diagnostics = [line.strip() for line in output.splitlines() if "Unhandled TEX conversion" in line]
    if not diagnostics:
        print("MyST reported no unhandled TeX nodes in the normalized chapter.")
        return

    kinds: Counter[str] = Counter()
    for line in diagnostics:
        match = re.search(r'node of "([^"]+)"', line)
        kinds[match.group(1) if match else "unknown"] += 1

    summary = ", ".join(f"{kind}: {count}" for kind, count in kinds.most_common())
    print(f"MyST reported {len(diagnostics)} unhandled TeX nodes ({summary}).")


def run_myst_isolated(normalized: str) -> tuple[str, str]:
    """Convert only the normalized TeX file in an isolated temporary directory."""
    with tempfile.TemporaryDirectory(prefix="bdl-myst-") as temp_dir_name:
        work_dir = Path(temp_dir_name)
        tex_file = work_dir / "chapter.tex"
        tex_file.write_text(normalized, encoding="utf-8")

        # Preserve repository-relative figure paths for the TeX-to-Markdown pass
        # without exposing MyST to the original source files as project inputs.
        (work_dir / "sampling_methods").symlink_to(TEX_ROOT.resolve() / "sampling_methods")

        log_path = work_dir / "myst.log"
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    ["myst", "build", tex_file.name, "--md"],
                    cwd=work_dir,
                    check=False,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=MYST_TIMEOUT_SECONDS,
                )
        except subprocess.TimeoutExpired as exc:
            output = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            summarize_myst_output(output)
            tail = "\n".join(output.splitlines()[-20:])
            raise RuntimeError(
                f"MyST conversion exceeded {MYST_TIMEOUT_SECONDS} seconds. "
                f"Final diagnostics:\n{tail}"
            ) from exc

        output = log_path.read_text(encoding="utf-8")
        summarize_myst_output(output)

        if result.returncode != 0:
            tail = "\n".join(output.splitlines()[-20:])
            raise RuntimeError(f"MyST conversion failed. Final diagnostics:\n{tail}")

        export_path = work_dir / "_build" / "exports" / "chapter.md"
        if not export_path.exists():
            raise FileNotFoundError(f"MyST did not create the expected Markdown export: {export_path}")

        return export_path.read_text(encoding="utf-8"), output


def copy_and_rewrite_images(markdown: str, source_dir: Path, output_path: Path) -> str:
    """Copy chapter figures into the book and rewrite generated image paths.

    When the TeX source references a PDF, prefer a same-stem web image if the
    source repository provides one. This avoids runtime PDF rasterization.
    """
    source_fig_dir = source_dir / "fig"
    target_fig_dir = output_path.parent / "assets" / "intro"
    target_fig_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(re.escape(SOURCE_IMAGE_PREFIX) + r"([^\s)\]}]+)")

    def replace(match: re.Match[str]) -> str:
        requested_name = match.group(1)
        requested = source_fig_dir / requested_name
        chosen = requested

        if requested.suffix.lower() == ".pdf":
            for extension in WEB_IMAGE_EXTENSIONS:
                candidate = requested.with_suffix(extension)
                if candidate.exists():
                    chosen = candidate
                    break

        if not chosen.exists():
            return match.group(0)

        destination = target_fig_dir / chosen.name
        shutil.copy2(chosen, destination)
        return f"assets/intro/{chosen.name}"

    return pattern.sub(replace, markdown)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=tex_path("sampling_methods", "intro", "main.tex"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("content/sampling_methods/intro.md"),
        help="Chapter 1 output used by the normal book TOC.",
    )
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    normalized, structures = prepare_latex(source)
    generated, _ = run_myst_isolated(normalized)

    markdown = clean_generated_markdown(generated, structures)
    markdown = copy_and_rewrite_images(markdown, args.input.parent, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
