#!/usr/bin/env python3
"""Build chapter PDFs before publishing HTML with native MyST download menus."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from chapter_downloads import chapter_slug, configure_downloads
from pdf_export import (
    export_markdown,
    normalize_math_paragraphs,
    repair_figures,
    repair_references,
    site_references,
)

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
TECTONIC_VERSION = "0.17.0"
TECTONIC_SHA256 = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"
TECTONIC_URL = (
    "https://github.com/tectonic-typesetting/tectonic/releases/download/"
    "tectonic%400.17.0/tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz"
)


def ensure_tectonic() -> Path:
    """Use an installed compiler or cache a verified official Linux release."""
    installed = shutil.which("tectonic")
    if installed:
        return Path(installed)
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("Install Tectonic (https://tectonic-typesetting.github.io/) first.")
    target = ROOT / "_build/tools" / f"tectonic-{TECTONIC_VERSION}" / "tectonic"
    if not target.exists():
        print(f"Downloading official Tectonic {TECTONIC_VERSION} compiler...", flush=True)
        with urllib.request.urlopen(TECTONIC_URL, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != TECTONIC_SHA256:
            raise RuntimeError("Tectonic archive checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            member = next(m for m in archive.getmembers() if Path(m.name).name == "tectonic")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("Tectonic binary missing from archive")
            binary = source.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(binary)
        target.chmod(0o755)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chapter", type=Path, help="Only build this content-relative chapter PDF")
    parser.add_argument("--pdf-only", action="store_true", help="Skip the final static HTML build")
    args = parser.parse_args()
    os.chdir(ROOT)
    all_chapters = configure_downloads()
    chapters = all_chapters
    if args.chapter:
        requested = (CONTENT / args.chapter).resolve()
        chapters = [(p, t) for p, t in chapters if p.resolve() == requested]
        if not chapters:
            parser.error("--chapter must name a migrated chapter relative to content/")
    compiler = ensure_tectonic()
    output = CONTENT / "_build/exports/chapters"
    output.mkdir(parents=True, exist_ok=True)
    pages = [p.relative_to(Path("content")).as_posix() for p, _ in chapters]
    workspace = ROOT / "_build/pdf-project"
    if workspace.exists():
        shutil.rmtree(workspace)
    source_copy = workspace / "content"
    shutil.copytree(CONTENT, source_copy, ignore=shutil.ignore_patterns("_build", "__pycache__"))
    shutil.copy2(ROOT / "references.bib", workspace / "references.bib")
    originals = {}
    for path, _ in all_chapters:
        copy = source_copy / path.relative_to(Path("content"))
        original = copy.read_text(encoding="utf-8")
        originals[copy] = original
        header, body = original.split("---\n\n", 1)
        header = re.sub(r"(?m)^downloads:\n(?:[ \t]+[^\n]*\n)*", "", header)
        copy.write_text(header + "---\n\n" + body, encoding="utf-8")
    # Resolve canonical online equation numbers before the PDF-only adaptations.
    # Downloads are omitted in this temporary pass because PDFs do not exist yet.
    subprocess.run([sys.executable, "-m", "jupyter_book", "build", "--site"],
                   cwd=source_copy, check=True)
    references = site_references(source_copy / "_build/site")
    for copy, original in originals.items():
        copy.write_text(original, encoding="utf-8")
    for page in pages:
        path = source_copy / page
        path.write_text(export_markdown(path.read_text(encoding="utf-8")), encoding="utf-8")
    # MyST's native TeX exporter preserves the exact LaTeX math and project
    # macros. Tectonic supplies TeX/BibTeX without a system TeX installation,
    # including on the Linux hosting builder. It fails on compilation errors.
    subprocess.run([sys.executable, "-m", "jupyter_book", "build", *pages, "--tex", "--strict"],
                   cwd=source_copy, check=True)
    for path, _ in chapters:
        slug = chapter_slug(path)
        tex_dir = source_copy / "_build/tex" / slug
        tex = tex_dir / f"{slug}.tex"
        if not tex.is_file():
            raise RuntimeError(f"MyST did not export fresh TeX for {path}")
        tex.write_text(repair_references(normalize_math_paragraphs(repair_figures(
            tex.read_text(encoding="utf-8"), path.read_text(encoding="utf-8")
        )), references), encoding="utf-8")
        for bib in tex_dir.glob("*.bib"):
            # Preserve the authoritative spelling/accents: citation-js 1.10
            # can corrupt TeX-encoded names when reserializing a bibliography.
            shutil.copy2(ROOT / "references.bib", bib)
        print(f"Building chapter PDF: {slug}", flush=True)
        subprocess.run([str(compiler), "--keep-logs", "--outdir", str(output), f"{slug}.tex"],
                       cwd=tex_dir, check=True)
        if not (output / f"{slug}.pdf").is_file():
            raise RuntimeError(f"PDF was not produced for {path}")
    if not args.pdf_only:
        subprocess.run([sys.executable, "-m", "jupyter_book", "build", "--html"],
                       cwd=CONTENT, check=True)


if __name__ == "__main__":
    main()
