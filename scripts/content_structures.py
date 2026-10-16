"""Translate source callouts and grouped equations to native MyST structures."""

from __future__ import annotations

import re
from dataclasses import dataclass

from latex_normalize import normalize_notation
from myst_structures import ExtractedStructure, _clean_inline_tex, parse_braced


@dataclass(frozen=True)
class CalloutStructure(ExtractedStructure):
    title: str
    body_tex: str


def extract_callouts(text: str) -> tuple[str, list[CalloutStructure]]:
    """Extract balanced callouts, retaining their bodies for recursive conversion."""
    structures: list[CalloutStructure] = []
    boundary = re.compile(r"\\(begin|end)\{callout\}")

    def extract(value: str) -> str:
        pieces: list[str] = []
        pos = 0
        while match := boundary.search(value, pos):
            if match.group(1) != "begin":
                raise ValueError("Unmatched end of callout")
            title, body_start = parse_braced(value, match.end())
            depth = 1
            for end in boundary.finditer(value, body_start):
                depth += 1 if end.group(1) == "begin" else -1
                if depth == 0:
                    break
            else:
                raise ValueError("Unclosed callout")
            body = extract(value[body_start:end.start()])
            token = f"BDLCALLOUTPLACEHOLDER{len(structures):04d}"
            structures.append(CalloutStructure(token, "", _clean_inline_tex(title), body))
            pieces.extend([value[pos:match.start()], f"\n\n{token}\n\n"])
            pos = end.end()
        pieces.append(value[pos:])
        return "".join(pieces)

    return extract(text), structures


def callout_directive(structure: CalloutStructure, body: str) -> str:
    """Use a longer native fence when the body contains nested directives."""
    inner_fences = [len(m.group(1)) for m in re.finditer(r"^(:{3,})", body, re.MULTILINE)]
    fence = ":" * max(3, max(inner_fences, default=0) + 1)
    return f"{fence}{{admonition}} {structure.title}\n\n{body.strip()}\n{fence}"


def _split_math_rows(text: str) -> list[str]:
    """Split outer row breaks, respecting brace groups and nested environments."""
    rows: list[str] = []
    depth = 0
    environments = 0
    pos = start = 0
    while pos < len(text):
        if text[pos] == "\\":
            environment = re.match(r"\\(begin|end)\{[^{}]+\}", text[pos:])
            if environment:
                environments += 1 if environment.group(1) == "begin" else -1
                pos += environment.end()
                continue
            if text.startswith(r"\\", pos) and depth == 0 and environments == 0:
                rows.append(text[start:pos])
                pos += 2
                spacing = re.match(r"\s*\[[^\]]*\]", text[pos:])
                if spacing:
                    pos += spacing.end()
                start = pos
                continue
            # Skip escaped braces and all other control sequences.
            command = re.match(r"\\(?:[A-Za-z]+|.)", text[pos:])
            if command:
                pos += command.end()
                continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    rows.append(text[start:])
    return [row.strip() for row in rows if row.strip()]


def extract_subequations(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Give each aligned subequation its own labelled, auto-numbered math block.

    A missing left-hand side inherits the preceding one, making continuation
    rows meaningful when displayed as separate equations. Unsupported groups
    fail explicitly rather than silently losing expressions or labels.
    """
    structures: list[ExtractedStructure] = []

    def replace(match: re.Match[str]) -> str:
        group = match.group(1)
        align = re.fullmatch(r"\s*\\begin\{align\*?\}(.*?)\\end\{align\*?\}\s*", group, re.DOTALL)
        if align is None:
            raise ValueError("Expected one align environment inside subequations")
        labels = re.findall(r"\\label\{([^{}]+)\}", group)
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate labels inside subequations")
        lhs = ""
        tokens: list[str] = []
        preserved: list[str] = []
        for row in _split_math_rows(align.group(1)):
            row_labels = re.findall(r"\\label\{([^{}]+)\}", row)
            if len(row_labels) > 1:
                raise ValueError("Multiple labels on a single subequation row")
            preserved.extend(row_labels)
            row = re.sub(r"\\label\{[^{}]+\}", "", row).strip()
            if "&" in row:
                left, right = row.split("&", 1)
                if left.strip():
                    lhs = left.strip()
                row = f"{lhs} {right.strip()}"
            row = normalize_notation(row)
            options = f":label: {row_labels[0]}\n\n" if row_labels else ""
            markdown = f"```{{math}}\n{options}{row.strip()}\n```"
            token = f"BDLMATHPLACEHOLDER{len(structures):04d}"
            structures.append(ExtractedStructure(token, markdown))
            tokens.append(token)
        if preserved != labels:
            raise ValueError("Subequation conversion did not preserve all labels")
        return "\n\n" + "\n\n".join(tokens) + "\n\n"

    pattern = r"\\begin\{subequations\}(.*?)\\end\{subequations\}"
    return re.sub(pattern, replace, text, flags=re.DOTALL), structures
