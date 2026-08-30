#!/usr/bin/env python3
"""Generate placeholder Part/chapter pages and the full MyST book TOC from TeX."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from common import tex_path


TOC_BEGIN = "    # BEGIN GENERATED BOOK TOC"
TOC_END = "    # END GENERATED BOOK TOC"
MYST_CONFIG = Path("myst.yml")
CONTENT_ROOT = Path("content")
PARKED_CHAPTERS: dict[Path, Path] = {
    Path("content/sampling_methods/intro.md"): Path("content/sampling_methods/intro_working.md"),
}


@dataclass(frozen=True)
class Chapter:
    source: str
    output: Path
    title: str
    label: str


@dataclass(frozen=True)
class Part:
    number: int
    title: str
    label: str
    output: Path
    chapters: tuple[Chapter, ...]


def parse_braced_argument(text: str, brace_start: int) -> tuple[str, int]:
    """Return a balanced braced argument and the index immediately after it."""
    if brace_start >= len(text) or text[brace_start] != "{":
        raise ValueError(f"Expected '{{' at position {brace_start}")

    depth = 0
    for pos in range(brace_start, len(text)):
        char = text[pos]
        escaped = pos > 0 and text[pos - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
        elif char == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos], pos + 1
    raise ValueError("Unclosed braced argument")


def command_argument(text: str, command: str) -> tuple[str, int] | None:
    """Extract the first required argument of a LaTeX command."""
    start = text.find(command)
    if start < 0:
        return None
    pos = start + len(command)
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos < len(text) and text[pos] == "[":
        end = text.find("]", pos + 1)
        if end < 0:
            raise ValueError(f"Unclosed optional argument for {command}")
        pos = end + 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None
    return parse_braced_argument(text, pos)


def clean_title(title: str) -> str:
    """Apply conservative title cleanup suitable for placeholder headings."""
    title = re.sub(r"\\texorpdfstring\{([^{}]*)\}\{([^{}]*)\}", r"\1", title)
    title = re.sub(r"\\text(?:bf|it|tt)\{([^{}]*)\}", r"\1", title)
    title = title.replace(r"\&", "&")
    title = title.replace("~", " ")
    title = re.sub(r"\s+", " ", title).strip()
    return title


def chapter_metadata(source: str) -> Chapter:
    path = tex_path(*source.split("/"))
    text = path.read_text(encoding="utf-8")

    chapter_arg = command_argument(text, r"\chapter")
    if chapter_arg is None:
        raise ValueError(f"Could not find \\chapter in {path}")
    raw_title, _ = chapter_arg

    labels = re.findall(r"\\label\{([^{}]+)\}", text)
    chapter_labels = [label for label in labels if label.startswith(("chap:", "cha:"))]
    if not chapter_labels:
        raise ValueError(f"Could not find a chapter label in {path}")

    source_path = Path(source)
    output = CONTENT_ROOT / source_path.parent.with_suffix(".md")
    return Chapter(
        source=source,
        output=output,
        title=clean_title(raw_title),
        label=chapter_labels[0],
    )


def parse_book_structure() -> tuple[Part, ...]:
    """Parse Parts and chapter inputs from the authoritative TeX main file."""
    text = tex_path("main.tex").read_text(encoding="utf-8")
    mainmatter = text.split(r"\mainmatter", 1)[1]
    bibliography = mainmatter.find(r"\bibliographystyle")
    if bibliography >= 0:
        mainmatter = mainmatter[:bibliography]

    part_matches = list(re.finditer(r"\\part\{([^{}]+)\}", mainmatter))
    parts: list[Part] = []

    for index, match in enumerate(part_matches, start=1):
        block_end = part_matches[index].start() if index < len(part_matches) else len(mainmatter)
        block = mainmatter[match.end() : block_end]

        label_match = re.search(r"\\label\{(part:[^{}]+)\}", block)
        if not label_match:
            raise ValueError(f"Could not find label for Part {index}: {match.group(1)}")
        label = label_match.group(1)

        chapter_sources = re.findall(r"\\input\{([^{}]+/main\.tex)\}", block)
        chapters = tuple(chapter_metadata(source) for source in chapter_sources)
        slug = label.removeprefix("part:")

        parts.append(
            Part(
                number=index,
                title=clean_title(match.group(1)),
                label=label,
                output=CONTENT_ROOT / "parts" / f"{slug}.md",
                chapters=chapters,
            )
        )

    return tuple(parts)


def placeholder_text(label: str, title: str, kind: str) -> str:
    return f"({label})=\n# {title}\n\n_{kind} migration pending._\n"


def write_placeholder(path: Path, label: str, title: str, kind: str) -> bool:
    """Create a placeholder page only when the destination does not already exist."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(placeholder_text(label, title, kind), encoding="utf-8")
    return True


def park_and_placeholder(path: Path, parked_path: Path, label: str, title: str) -> bool:
    """Preserve an existing migrated page and replace it with a chapter placeholder."""
    path.parent.mkdir(parents=True, exist_ok=True)

    expected_placeholder = placeholder_text(label, title, "Chapter")
    if path.exists() and path.read_text(encoding="utf-8") != expected_placeholder:
        if not parked_path.exists():
            parked_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, parked_path)
            print(f"Parked {path} -> {parked_path}")

    changed = not path.exists() or path.read_text(encoding="utf-8") != expected_placeholder
    if changed:
        path.write_text(expected_placeholder, encoding="utf-8")
    return changed


def render_toc(parts: tuple[Part, ...]) -> str:
    lines: list[str] = []
    for part in parts:
        lines.append(f"    - file: {part.output.as_posix()}")
        lines.append("      children:")
        for chapter in part.chapters:
            lines.append(f"        - file: {chapter.output.as_posix()}")
    return "\n".join(lines)


def update_myst_toc(parts: tuple[Part, ...]) -> None:
    text = MYST_CONFIG.read_text(encoding="utf-8")
    if TOC_BEGIN not in text or TOC_END not in text:
        raise ValueError(
            f"{MYST_CONFIG} must contain the generated TOC markers before running this script"
        )

    before, rest = text.split(TOC_BEGIN, 1)
    _, after = rest.split(TOC_END, 1)
    generated = render_toc(parts)
    MYST_CONFIG.write_text(
        f"{before}{TOC_BEGIN}\n{generated}\n{TOC_END}{after}",
        encoding="utf-8",
    )


def main() -> None:
    parts = parse_book_structure()
    created = 0

    for part in parts:
        if write_placeholder(
            part.output,
            part.label,
            f"Part {part.number} — {part.title}",
            "Part",
        ):
            created += 1
        for chapter in part.chapters:
            parked_path = PARKED_CHAPTERS.get(chapter.output)
            if parked_path is not None:
                if park_and_placeholder(
                    chapter.output,
                    parked_path,
                    chapter.label,
                    chapter.title,
                ):
                    created += 1
            elif write_placeholder(chapter.output, chapter.label, chapter.title, "Chapter"):
                created += 1

    update_myst_toc(parts)
    chapter_count = sum(len(part.chapters) for part in parts)
    print(
        f"Book skeleton: {len(parts)} parts, {chapter_count} chapters; "
        f"created/updated {created} placeholder pages."
    )


if __name__ == "__main__":
    main()
