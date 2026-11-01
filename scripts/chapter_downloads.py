"""Native MyST exports and labelled downloads for migrated chapters."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

CONTENT_ROOT = Path("content")


def chapter_slug(path: Path) -> str:
    relative = path.resolve().relative_to(CONTENT_ROOT.resolve()).with_suffix("")
    return "-".join(relative.parts).replace("_", "-")


def with_chapter_downloads(markdown: str, path: Path, title: str) -> str:
    """Add generated export metadata without changing the chapter body."""
    # Diagnostic conversions can target arbitrary temporary directories. They
    # should not acquire downloads belonging to the published book.
    if not path.resolve().is_relative_to(CONTENT_ROOT.resolve()):
        return markdown
    if markdown.startswith("---\n"):
        markdown = re.sub(r"\A---\n.*?\n---\n\n?", "", markdown, count=1, flags=re.S)
    slug = chapter_slug(path)
    template = Path(os.path.relpath(CONTENT_ROOT / "_templates/chapter", path.parent)).as_posix()
    pdf = Path(os.path.relpath(
        CONTENT_ROOT / "_build/exports/chapters" / f"{slug}.pdf", path.parent
    )).as_posix()
    tex = Path(os.path.relpath(
        CONTENT_ROOT / "_build/tex" / slug / f"{slug}.tex", path.parent
    )).as_posix()
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        "exports:\n"
        "  - format: pdf\n"
        f"    template: {template}\n"
        f"    output: {pdf}\n"
        f"    id: pdf-{slug}\n"
        "  - format: tex\n"
        f"    template: {template}\n"
        f"    output: {tex}\n"
        "downloads:\n"
        f"  - id: pdf-{slug}\n"
        "    title: Download chapter PDF\n"
        f"  - file: {path.name}\n"
        "    title: Download Markdown source\n"
        "---\n\n"
        + markdown
    )


def migrated_chapters() -> list[tuple[Path, str]]:
    # Hosting builds only need the checked-in MyST repository, not its sibling
    # TeX checkout. The generated TOC supplies chapter order; the canonical
    # heading supplies the same title displayed online.
    config = (CONTENT_ROOT / "myst.yml").read_text(encoding="utf-8")
    chapters = []
    for filename in re.findall(r"(?m)^\s*- file: (\S+\.md)\s*$", config):
        path = CONTENT_ROOT / filename
        markdown = path.read_text(encoding="utf-8")
        if "_Chapter migration pending._" in markdown:
            continue
        heading = re.search(r"(?m)^\((?:chap|cha):[^\n]+\)=\n# ([^\n]+)", markdown)
        if heading:
            chapters.append((path, heading.group(1)))
    return chapters


def configure_downloads() -> list[tuple[Path, str]]:
    chapters = migrated_chapters()
    for path, title in chapters:
        original = path.read_text(encoding="utf-8")
        updated = with_chapter_downloads(original, path, title)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
    return chapters


if __name__ == "__main__":
    print(f"Configured native downloads for {len(configure_downloads())} migrated chapters.")
