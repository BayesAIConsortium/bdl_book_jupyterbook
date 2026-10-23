#!/usr/bin/env python3
"""Reusable deterministic conversion of TeX footnotes to MyST Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass

from latex_normalize import normalize_latex
from semantic_references import extract_references


@dataclass(frozen=True)
class FootnoteStructure:
    """One extracted TeX footnote and its deterministic Markdown identity."""

    placeholder: str
    identifier: str
    body: str


def _parse_braced(text: str, start: int) -> tuple[str, int]:
    """Parse one balanced braced argument and return its contents and next index."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError(f"Expected '{{' at position {start}")

    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif text[pos] == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos], pos + 1
    raise ValueError("Unclosed brace group")


def _citation_keys(raw: str) -> list[str]:
    return [key.strip() for key in raw.split(",") if key.strip()]


def _clean_body(text: str) -> str:
    """Convert the conservative inline TeX subset used in book footnotes."""
    text = normalize_latex(text)
    text, references = extract_references(text)
    text = re.sub(
        r"\\(?:cite|citep)\{([^{}]+)\}",
        lambda match: "[" + "; ".join(
            f"@{key}" for key in _citation_keys(match.group(1))
        ) + "]",
        text,
    )
    text = re.sub(
        r"\\citet\{([^{}]+)\}",
        lambda match: (
            f"@{_citation_keys(match.group(1))[0]}"
            if len(_citation_keys(match.group(1))) == 1
            else "[" + "; ".join(
                f"@{key}" for key in _citation_keys(match.group(1))
            ) + "]"
        ),
        text,
    )
    text = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"`\1`", text)
    text = text.replace(r"\&", "&")
    text = text.replace("~", " ")
    text = text.replace("---", "—")
    text = text.replace("--", "–")
    for reference in references:
        text = text.replace(reference.placeholder, reference.markdown)
    return re.sub(r"\s+", " ", text).strip()


def extract_footnotes(text: str) -> tuple[str, list[FootnoteStructure]]:
    """Replace TeX footnotes with opaque placeholders in deterministic order."""
    structures: list[FootnoteStructure] = []
    pieces: list[str] = []
    pos = 0

    while True:
        command_pos = text.find(r"\footnote", pos)
        if command_pos < 0:
            pieces.append(text[pos:])
            break

        after_command = command_pos + len(r"\footnote")
        if after_command < len(text) and text[after_command].isalpha():
            pieces.append(text[pos:after_command])
            pos = after_command
            continue

        pieces.append(text[pos:command_pos])
        body, end = _parse_braced(text, after_command)
        index = len(structures) + 1
        placeholder = f"BDLFOOTNOTEPLACEHOLDER{index:04d}"
        identifier = f"footnote-{index}"
        structures.append(
            FootnoteStructure(
                placeholder=placeholder,
                identifier=identifier,
                body=_clean_body(body),
            )
        )
        pieces.append(placeholder)
        pos = end

    return "".join(pieces), structures


def restore_footnotes(text: str, structures: list[FootnoteStructure]) -> str:
    """Restore footnote references and append deterministic Markdown definitions."""
    if not structures:
        return text

    definitions: list[str] = []
    for structure in structures:
        if structure.placeholder not in text:
            raise ValueError(
                f"Could not restore footnote placeholder {structure.placeholder}"
            )
        text = text.replace(
            structure.placeholder,
            f"[^{structure.identifier}]",
            1,
        )
        definitions.append(f"[^{structure.identifier}]: {structure.body}")

    if "BDLFOOTNOTEPLACEHOLDER" in text:
        raise ValueError("Unrestored footnote placeholder remained in generated Markdown")

    return text.rstrip() + "\n\n" + "\n\n".join(definitions) + "\n"
