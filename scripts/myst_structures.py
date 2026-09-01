#!/usr/bin/env python3
"""Reusable semantic conversion of unsupported LaTeX structures to native MyST."""

from __future__ import annotations

import re
from dataclasses import dataclass

from latex_normalize import normalize_notation, strip_tex_comments


PROOF_KINDS = (
    "example",
    "proposition",
    "definition",
    "lemma",
    "theorem",
    "remark",
    "assumption",
    "proof",
)


@dataclass(frozen=True)
class ExtractedStructure:
    placeholder: str
    markdown: str


def parse_braced(text: str, start: int) -> tuple[str, int]:
    """Parse one balanced {...} argument, returning its contents and next index."""
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


def convert_latex_references(text: str) -> str:
    """Translate common LaTeX cross-references to MyST internal references.

    The original TeX labels are retained as MyST target identifiers, so these
    links can resolve immediately against scaffold placeholder pages and remain
    valid when those placeholders are later replaced by migrated content.
    """
    text = re.sub(
        r"\b(Part|Chapter|Section|Figure|Table|Equation)~\\(?:ref|autoref)\{([^{}]+)\}",
        lambda match: f"{match.group(1)} [](#${match.group(2)})".replace("#$", "#"),
        text,
    )
    text = re.sub(
        r"\\eqref\{([^{}]+)\}",
        lambda match: f"[](#${match.group(1)})".replace("#$", "#"),
        text,
    )
    text = re.sub(
        r"\\(?:ref|autoref)\{([^{}]+)\}",
        lambda match: f"[](#${match.group(1)})".replace("#$", "#"),
        text,
    )
    return text


def _clean_algorithm_fragment(text: str) -> str:
    text = normalize_notation(text)
    text = text.replace(r"\KwTo", " to ")
    text = text.replace(r"\;", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _algorithm_sequence(text: str, indent: int = 0) -> list[str]:
    """Translate the algorithm2e subset used by the book to Markdown steps."""
    lines: list[str] = []
    pos = 0
    prefix = "  " * indent

    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue

        matched = False
        for command, label in (
            (r"\Input", "**Inputs**"),
            (r"\Output", "**Output**"),
            (r"\Return", "**Return**"),
            (r"\tcp", "*Note*"),
        ):
            if text.startswith(command, pos):
                arg, pos = parse_braced(text, pos + len(command))
                lines.append(f"{prefix}{label} {_clean_algorithm_fragment(arg)}")
                matched = True
                break
        if matched:
            continue

        for command, label in ((r"\For", "For"), (r"\While", "While"), (r"\If", "If")):
            if text.startswith(command, pos):
                condition, next_pos = parse_braced(text, pos + len(command))
                body, pos = parse_braced(text, next_pos)
                lines.append(f"{prefix}1. **{label}** {_clean_algorithm_fragment(condition)}:")
                lines.extend(_algorithm_sequence(body, indent + 1))
                matched = True
                break
        if matched:
            continue

        if text.startswith(r"\eIf", pos):
            condition, next_pos = parse_braced(text, pos + len(r"\eIf"))
            yes_body, next_pos = parse_braced(text, next_pos)
            no_body, pos = parse_braced(text, next_pos)
            lines.append(f"{prefix}1. **If** {_clean_algorithm_fragment(condition)}:")
            lines.extend(_algorithm_sequence(yes_body, indent + 1))
            lines.append(f"{prefix}2. **Else:**")
            lines.extend(_algorithm_sequence(no_body, indent + 1))
            continue

        semicolon = text.find(r"\;", pos)
        newline = text.find("\n", pos)
        end_candidates = [idx for idx in (semicolon, newline) if idx >= 0]
        end = min(end_candidates) if end_candidates else len(text)
        fragment = _clean_algorithm_fragment(text[pos:end])
        if fragment:
            lines.append(f"{prefix}1. {fragment}")
        if end >= len(text):
            pos = len(text)
        elif text.startswith(r"\;", end):
            pos = end + 2
        else:
            pos = end + 1

    return lines


def algorithm_to_myst(body: str) -> str:
    """Convert one algorithm2e environment body to MyST's native algorithm proof directive."""
    caption_match = re.search(r"\\caption\{([^{}]+)\}", body)
    label_match = re.search(r"\\label\{([^{}]+)\}", body)
    caption = caption_match.group(1).strip() if caption_match else "Algorithm"
    label = label_match.group(1).strip() if label_match else None

    body = re.sub(r"\\caption\{[^{}]+\}\s*", "", body, count=1)
    body = re.sub(r"\\label\{[^{}]+\}\s*", "", body, count=1)
    body = re.sub(
        r"^\s*\\SetKwInOut\{[^{}]+\}\{[^{}]+\}\s*$",
        "",
        body,
        flags=re.MULTILINE,
    )
    body = strip_tex_comments(body)

    lines = [f":::{{prf:algorithm}} {caption}"]
    if label:
        lines.append(f":label: {label}")
    lines.append("")
    lines.extend(_algorithm_sequence(body))
    lines.append(":::")
    return "\n".join(lines)


def extract_algorithms(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Replace active algorithm2e environments with placeholders and MyST versions.

    Strip TeX comments before semantic extraction so legacy environments that
    are fully commented out cannot be mistaken for active algorithms. This also
    ensures later theorem/proof extraction sees the same active source only.
    """
    text = strip_tex_comments(text)
    structures: list[ExtractedStructure] = []
    pattern = re.compile(
        r"\\begin\{algorithm\}(?:\[[^\]]*\])?(.*?)\\end\{algorithm\}",
        re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        placeholder = f"BDLALGORITHMPLACEHOLDER{len(structures):04d}"
        structures.append(ExtractedStructure(placeholder, algorithm_to_myst(match.group(1))))
        return f"\n\n{placeholder}\n\n"

    return pattern.sub(replace, text), structures


def mark_proof_environments(text: str) -> str:
    """Mark theorem-like boundaries so MyST can convert the bodies as ordinary content."""
    for kind in PROOF_KINDS:
        pattern = re.compile(rf"\\begin\{{{kind}\}}(.*?)\\end\{{{kind}\}}", re.DOTALL)

        def replace(match: re.Match[str], proof_kind: str = kind) -> str:
            body = match.group(1)
            label_match = re.match(r"\s*\\label\{([^{}]+)\}\s*", body)
            label = label_match.group(1) if label_match else ""
            if label_match:
                body = body[label_match.end() :]
            begin = f"BDLPROOFBEGIN {proof_kind} {label}".rstrip()
            end = f"BDLPROOFEND {proof_kind}"
            return f"\n\n{begin}\n\n{body.strip()}\n\n{end}\n\n"

        text = pattern.sub(replace, text)
    return text


def restore_proof_directives(text: str) -> str:
    """Turn semantic proof markers into MyST's native proof directives."""
    pattern = re.compile(
        r"BDLPROOFBEGIN\s+(example|proposition|definition|lemma|theorem|remark|assumption|proof)(?:\s+([^\s]+))?\s*\n(.*?)\n\s*BDLPROOFEND\s+\1",
        flags=re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        kind = match.group(1)
        label = match.group(2)
        body = match.group(3).strip()
        directive = "proof" if kind == "proof" else f"prf:{kind}"
        lines = [f":::{{{directive}}}"]
        if label:
            lines.append(f":label: {label}")
        lines.append("")
        lines.append(body)
        lines.append(":::")
        return "\n".join(lines)

    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(replace, text)
    return text


def normalize_headings_for_latex_pass(text: str) -> str:
    """Map a chapter heading to a standalone heading for MyST's LaTeX conversion pass.

    The chapter-level label is intentionally removed here. Individual chapter
    converters add one canonical page target after the isolated conversion,
    which avoids duplicate page identifiers while preserving all lower-level
    semantic labels in the converted content.
    """
    text = re.sub(
        r"\\chapter(?:\[[^\]]*\])?\{([^{}]+)\}\s*(?:\\label\{(?:chap|cha):[^{}]+\}\s*)?",
        lambda match: f"\\section*{{{match.group(1)}}}\n",
        text,
        count=1,
    )
    return text


def restore_extracted_structures(text: str, structures: list[ExtractedStructure]) -> str:
    """Replace placeholders in converted Markdown with native MyST structures."""
    for structure in structures:
        text = text.replace(structure.placeholder, structure.markdown)
    return restore_proof_directives(text)
