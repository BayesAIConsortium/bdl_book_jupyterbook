#!/usr/bin/env python3
"""Convert the full LaTeX Reader's Guide to native MyST Markdown."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import tex_path
from generate_book_skeleton import parse_book_structure
from latex_normalize import normalize_notation


BOOK_PARTS = parse_book_structure()
PART_NUMBERS = {part.label: part.number for part in BOOK_PARTS}
CHAPTER_NUMBERS: dict[str, int] = {}
chapter_number = 1
for part in BOOK_PARTS:
    for chapter in part.chapters:
        CHAPTER_NUMBERS[chapter.label] = chapter_number
        chapter_number += 1

TABLE_NUMBERS = {
    "tab:readerguide:prereq-early": 1,
    "tab:readerguide:prereq-late": 2,
}
FIGURE_NUMBERS = {"fig:readerguide:partmap": 1}

SECTION_TITLES = (
    "What this book covers",
    "How the book is organised",
    "Six chapters that unlock the rest",
    "The ten parts at a glance",
    "A map of the parts",
    "Chapter-level prerequisites",
    "Suggested routes through the book",
)


def parse_braced(text: str, start: int) -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError(f"Expected '{{' at position {start}")

    depth = 0
    for pos in range(start, len(text)):
        escaped = pos > 0 and text[pos - 1] == "\\"
        if text[pos] == "{" and not escaped:
            depth += 1
        elif text[pos] == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos], pos + 1
    raise ValueError("Unclosed braced argument")


def command_argument(text: str, command: str, start: int = 0) -> tuple[str, int, int] | None:
    command_start = text.find(command, start)
    if command_start < 0:
        return None
    pos = command_start + len(command)
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text) or text[pos] != "{":
        return None
    value, end = parse_braced(text, pos)
    return value, command_start, end


def convert_part_ref(label: str) -> str:
    number = PART_NUMBERS.get(label)
    text = f"Part {number}" if number is not None else "Part"
    return f"[{text}](#{label})"


def convert_chapter_ref(label: str, prefix: str = "Chapter") -> str:
    number = CHAPTER_NUMBERS.get(label)
    text = f"{prefix} {number}" if number is not None else prefix
    return f"[{text}](#{label})"


def convert_bare_ref(label: str) -> str:
    if label in CHAPTER_NUMBERS:
        return f"[{CHAPTER_NUMBERS[label]}](#{label})"
    if label in PART_NUMBERS:
        return f"[{PART_NUMBERS[label]}](#{label})"
    if label in TABLE_NUMBERS:
        return f"[{TABLE_NUMBERS[label]}](#{label})"
    if label in FIGURE_NUMBERS:
        return f"[{FIGURE_NUMBERS[label]}](#{label})"
    return f"[link](#{label})"


def convert_references(text: str) -> str:
    """Translate Reader's Guide cross-references using scaffold numbering."""
    text = re.sub(
        r"Parts~\\ref\{(part:[^{}]+)\}--\\ref\{(part:[^{}]+)\}",
        lambda m: f"{convert_part_ref(m.group(1))}–{convert_part_ref(m.group(2))}",
        text,
    )
    text = re.sub(
        r"Chapters~\\ref\{([^{}]+)\}--\\ref\{([^{}]+)\}",
        lambda m: f"{convert_chapter_ref(m.group(1), 'Chapters')}–{convert_bare_ref(m.group(2))}",
        text,
    )
    text = re.sub(
        r"Chapters~\\ref\{([^{}]+)\}",
        lambda m: convert_chapter_ref(m.group(1), "Chapters"),
        text,
    )
    text = re.sub(
        r"Chapter~\\ref\{([^{}]+)\}",
        lambda m: convert_chapter_ref(m.group(1)),
        text,
    )
    text = re.sub(
        r"Part~\\ref\{(part:[^{}]+)\}",
        lambda m: convert_part_ref(m.group(1)),
        text,
    )
    text = re.sub(
        r"Tables~\\ref\{([^{}]+)\}\s+and~\\ref\{([^{}]+)\}",
        lambda m: f"[Tables {TABLE_NUMBERS.get(m.group(1), '?')}](#{m.group(1)}) and "
        f"[{TABLE_NUMBERS.get(m.group(2), '?')}](#{m.group(2)})",
        text,
    )
    text = re.sub(
        r"Table~\\ref\{([^{}]+)\}",
        lambda m: f"[Table {TABLE_NUMBERS.get(m.group(1), '?')}](#{m.group(1)})",
        text,
    )
    text = re.sub(
        r"Figure~\\ref\{([^{}]+)\}",
        lambda m: f"[Figure {FIGURE_NUMBERS.get(m.group(1), '?')}](#{m.group(1)})",
        text,
    )
    text = re.sub(r"\\ref\{([^{}]+)\}", lambda m: convert_bare_ref(m.group(1)), text)
    return text


def clean_inline(text: str) -> str:
    """Convert the inline TeX subset used by the Reader's Guide."""
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    text = text.replace("---", "—")
    text = text.replace("--", "–")
    text = normalize_notation(text)
    text = convert_references(text)
    text = re.sub(r"\\textit\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = text.replace(r"\noindent", "")
    text = text.replace(r"\small", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def section_body(text: str, title: str) -> str:
    marker = rf"\section*{{{title}}}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find section: {title}")
    body_start = start + len(marker)

    later = []
    for other in SECTION_TITLES:
        other_marker = rf"\section*{{{other}}}"
        pos = text.find(other_marker, body_start)
        if pos >= 0:
            later.append(pos)
    end = min(later) if later else len(text)
    return text[body_start:end]


def split_paragraph_blocks(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Split an intro followed by LaTeX paragraph headings, preserving nested braces."""
    starts: list[tuple[int, int, str]] = []
    pos = 0
    command = r"\paragraph"
    while True:
        found = body.find(command, pos)
        if found < 0:
            break
        arg_start = found + len(command)
        while arg_start < len(body) and body[arg_start].isspace():
            arg_start += 1
        if arg_start >= len(body) or body[arg_start] != "{":
            pos = found + len(command)
            continue
        raw_heading, end = parse_braced(body, arg_start)
        heading = raw_heading
        if heading.startswith(r"\textbf{"):
            inner_start = len(r"\textbf")
            heading, _ = parse_braced(heading, inner_start)
        starts.append((found, end, heading))
        pos = end

    if not starts:
        return clean_inline(body), []

    intro = clean_inline(body[: starts[0][0]])
    blocks: list[tuple[str, str]] = []
    for index, (start, heading_end, heading) in enumerate(starts):
        content_end = starts[index + 1][0] if index + 1 < len(starts) else len(body)
        blocks.append((clean_inline(heading), clean_inline(body[heading_end:content_end])))
    return intro, blocks


def render_paragraph_section(body: str) -> str:
    intro, blocks = split_paragraph_blocks(body)
    pieces = [intro] if intro else []
    for heading, paragraph in blocks:
        pieces.append(f"**{heading}** {paragraph}")
    return "\n\n".join(pieces)


def convert_itemize(text: str) -> str:
    match = re.search(r"\\begin\{itemize\}(.*?)\\end\{itemize\}", text, re.DOTALL)
    if not match:
        return clean_inline(text)

    before = clean_inline(text[: match.start()])
    after = clean_inline(text[match.end() :])
    raw_items = re.split(r"\\item\s+", match.group(1))[1:]
    items = "\n".join(f"- {clean_inline(item)}" for item in raw_items)
    return "\n\n".join(piece for piece in (before, items, after) if piece)


def split_table_rows(tabular: str) -> list[str]:
    tabular = re.sub(r"\\(?:toprule|midrule|bottomrule|addlinespace)\s*", "", tabular)
    return [row.strip() for row in re.split(r"\\\\\s*", tabular) if row.strip()]


def render_table(table_env: str) -> str:
    caption_info = command_argument(table_env, r"\caption")
    label_match = re.search(r"\\label\{([^{}]+)\}", table_env)
    tabular_match = re.search(r"\\begin\{tabularx\}\{.*?\}\{.*?\}(.*?)\\end\{tabularx\}", table_env, re.DOTALL)
    if caption_info is None or label_match is None or tabular_match is None:
        raise ValueError("Could not parse Reader's Guide prerequisite table")

    caption = clean_inline(caption_info[0])
    label = label_match.group(1)
    number = TABLE_NUMBERS[label]
    rows = split_table_rows(tabular_match.group(1))

    output = [f":::{{table}} Table {number}. {caption}", f":label: {label}", ""]
    header_written = False
    for row in rows:
        multicolumn = re.search(r"\\multicolumn\{4\}\{.*?\}\{(.*)\}\s*$", row, re.DOTALL)
        if multicolumn:
            group = clean_inline(multicolumn.group(1))
            output.extend([f"**{group}**", ""])
            header_written = False
            continue

        cells = [clean_inline(cell) for cell in re.split(r"(?<!\\)&", row)]
        if len(cells) != 4:
            continue
        cells = [cell.replace("|", r"\|") for cell in cells]
        output.append("| " + " | ".join(cells) + " |")
        if not header_written:
            output.append("| --- | --- | --- | --- |")
            header_written = True
    output.append(":::")
    return "\n".join(output)


def extract_tables(body: str) -> tuple[str, list[str]]:
    tables: list[str] = []
    pattern = re.compile(r"\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}", re.DOTALL)

    def replace(match: re.Match[str]) -> str:
        placeholder = f"BDLREADERTABLE{len(tables):02d}"
        tables.append(render_table(match.group(0)))
        return f"\n\n{placeholder}\n\n"

    return pattern.sub(replace, body), tables


def restore_tables(text: str, tables: list[str]) -> str:
    for index, table in enumerate(tables):
        text = text.replace(f"BDLREADERTABLE{index:02d}", table)
    return text


def render_part_map() -> str:
    """Render the source TikZ dependency map as a web-native Mermaid diagram."""
    return """(fig:readerguide:partmap)=
```mermaid
flowchart TB
  P1["I. Sampling methods<br/>Ch. 1–6<br/><i>enter at 1</i>"]
  P3["III. Variational inference<br/>Ch. 12–18<br/><i>enter at 12</i>"]
  P5["V. Kernel methods<br/>Ch. 23–27<br/><i>enter at 23</i>"]
  P2["II. Laplace approximations<br/>Ch. 7–11<br/><i>enter at 7</i>"]
  P4["IV. Ensemble methods<br/>Ch. 19–22<br/><i>enter at 19</i>"]
  P6["VI. Priors<br/>Ch. 28–31<br/><i>enter at 28</i>"]
  P7["VII. Identifiability and symmetries<br/>Ch. 32–36"]
  P8["VIII. Scalability<br/>Ch. 37–42"]
  P9["IX. Applications<br/>Ch. 43–47"]
  P10["X. Topical developments<br/>Ch. 48–56"]

  P1 --> P2
  P3 --> P4
  P5 --> P6
  P3 --> P6
  P2 --> P8
  P6 --> P8
  P6 --> P7
  P8 --> P9
  P8 --> P10
  P7 --> P10
  P5 --> P8
  P1 -. useful, not required .-> P7
```

*Figure 1. Recommended dependences between the ten parts. Solid arrows mark background the target part assumes; the dashed arrow marks material that enriches it without being required. The entry parts—Sampling methods, Variational inference, and Kernel methods—have no prerequisites and can be read first.*"""


def convert(text: str) -> str:
    callout = re.search(r"\\begin\{callout\}\{([^{}]+)\}(.*?)\\end\{callout\}", text, re.DOTALL)
    if not callout:
        raise ValueError("Could not find Reader's Guide callout")

    pieces = [
        "# Reader's guide",
        f":::{'{'}note{'}'} {clean_inline(callout.group(1))}\n{clean_inline(callout.group(2))}\n:::",
        "## What this book covers\n\n" + clean_inline(section_body(text, "What this book covers")),
        "## How the book is organised\n\n" + render_paragraph_section(section_body(text, "How the book is organised")),
        "## Six chapters that unlock the rest\n\n" + convert_itemize(section_body(text, "Six chapters that unlock the rest")),
        "## The ten parts at a glance\n\n" + render_paragraph_section(section_body(text, "The ten parts at a glance")),
    ]

    map_body = section_body(text, "A map of the parts")
    figure_match = re.search(r"\\begin\{figure\}.*?\\end\{figure\}", map_body, re.DOTALL)
    map_prose = clean_inline(map_body[: figure_match.start()] if figure_match else map_body)
    pieces.append("## A map of the parts\n\n" + map_prose + "\n\n" + render_part_map())

    prereq_body = section_body(text, "Chapter-level prerequisites")
    prereq_without_tables, tables = extract_tables(prereq_body)
    prereq_text = clean_inline(prereq_without_tables)
    prereq_text = restore_tables(prereq_text, tables)
    pieces.append("## Chapter-level prerequisites\n\n" + prereq_text)

    pieces.append(
        "## Suggested routes through the book\n\n"
        + render_paragraph_section(section_body(text, "Suggested routes through the book"))
    )

    return "\n\n".join(pieces).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=tex_path("frontmatter", "readerguide.tex"))
    parser.add_argument("--output", type=Path, default=Path("content/frontmatter/readers_guide.md"))
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(convert(source), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
