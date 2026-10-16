"""Shared parsing of editorial metadata from authoritative LaTeX sources."""

from __future__ import annotations

import re


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


def source_chapter_title(text: str) -> str | None:
    r"""Read the full ``\chapter{...}`` title, ignoring its optional short title."""
    text = re.sub(r"(?<!\\)%[^\n]*(?:\n|$)", "", text)
    match = re.search(r"\\chapter(?![A-Za-z])", text)
    if match is None:
        return None
    argument = command_argument(text[match.start():], r"\chapter")
    if argument is None:
        raise ValueError("Chapter command has no title argument")
    return clean_title(argument[0])
