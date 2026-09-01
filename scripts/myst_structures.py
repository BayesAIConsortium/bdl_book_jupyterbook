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


def _clean_inline_tex(text: str) -> str:
    """Convert the conservative inline TeX subset used in extracted captions."""
    text = normalize_notation(text)
    text = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = text.replace(r"\&", "&")
    text = text.replace("~", " ")
    text = text.replace("---", "—")
    text = text.replace("--", "–")
    return re.sub(r"\s+", " ", text).strip()


def _section_label_titles(text: str) -> dict[str, str]:
    """Map section-like labels to human-readable titles for robust web references."""
    titles: dict[str, str] = {}
    pattern = re.compile(
        r"\\(?:section|subsection|subsubsection)\*?\{([^{}]+)\}\s*\\label\{([^{}]+)\}",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        titles[match.group(2)] = _clean_inline_tex(match.group(1))
    return titles


def extract_references(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Protect semantic TeX references that need MyST-friendly display text.

    MyST's LaTeX converter can resolve many references directly, but section
    numbers are not necessarily enumerated in the web theme and can surface as
    ``??``. Equation references also lose the word ``Equation`` in some contexts.
    Protect chapter references as native MyST links as well, so their targets
    remain stable while placeholder chapters are replaced by migrated content.
    """
    structures: list[ExtractedStructure] = []
    section_titles = _section_label_titles(text)

    def placeholder(markdown: str) -> str:
        token = f"BDLREFERENCEPLACEHOLDER{len(structures):04d}"
        structures.append(ExtractedStructure(token, markdown))
        return token

    text = re.sub(
        r"\b([Ee]quation)\s*~?\s*\\eqref\{([^{}]+)\}",
        lambda m: placeholder(f"{m.group(1)} [](#{m.group(2)})"),
        text,
    )
    text = re.sub(
        r"\\eqref\{([^{}]+)\}",
        lambda m: placeholder(f"Equation [](#{m.group(1)})"),
        text,
    )
    text = re.sub(
        r"\bSection\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(
            f"Section [{section_titles.get(m.group(1), 'link')}](#{m.group(1)})"
        ),
        text,
    )
    text = re.sub(
        r"\b(Chapter|Chapters)\s*~?\s*\\ref\{([^{}]+)\}",
        lambda m: placeholder(f"{m.group(1)} [](#{m.group(2)})"),
        text,
    )
    return text, structures


def _clean_algorithm_fragment(text: str) -> str:
    """Normalize one algorithm2e text fragment to MyST-friendly Markdown."""
    text = normalize_notation(text)
    text = re.sub(
        r"\\eqref\{([^{}]+)\}",
        lambda match: f"Equation [](#${match.group(1)})".replace("#$", "#"),
        text,
    )
    text = re.sub(
        r"\\ref\{([^{}]+)\}",
        lambda match: f"[](#${match.group(1)})".replace("#$", "#"),
        text,
    )
    text = text.replace(r"\KwTo", " to ")
    text = text.replace(r"\;", "")
    return re.sub(r"\s+", " ", text).strip()


def _algorithm_sequence(text: str, indent: int = 0) -> list[str]:
    """Translate the algorithm2e subset used by the book to Markdown steps."""
    lines: list[str] = []
    pos = 0
    prefix = "    " * indent

    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue

        if text.startswith(r"\[", pos):
            end = text.find(r"\]", pos + 2)
            if end < 0:
                end = len(text)
                next_pos = end
            else:
                next_pos = end + 2
            math = normalize_notation(text[pos + 2 : end]).strip()
            lines.extend([f"{prefix}$$", math, f"{prefix}$$"])
            pos = next_pos
            if text.startswith(r"\;", pos):
                pos += 2
            continue

        matched = False
        for command, label in (
            (r"\Input", "Inputs"),
            (r"\Output", "Output"),
        ):
            if text.startswith(command, pos):
                arg, pos = parse_braced(text, pos + len(command))
                lines.append(f"{prefix}- **{label}:** {_clean_algorithm_fragment(arg)}")
                matched = True
                break
        if matched:
            continue

        if text.startswith(r"\Return", pos):
            arg, pos = parse_braced(text, pos + len(r"\Return"))
            lines.append(f"{prefix}1. **Return** {_clean_algorithm_fragment(arg)}")
            continue

        if text.startswith(r"\tcp", pos):
            arg, pos = parse_braced(text, pos + len(r"\tcp"))
            lines.append(f"{prefix}*Note:* {_clean_algorithm_fragment(arg)}")
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
    caption = _clean_inline_tex(caption_match.group(1)) if caption_match else "Algorithm"
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
    """Replace active algorithm2e environments with placeholders and MyST versions."""
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


def figure_to_myst(body: str) -> str:
    """Convert a TeX figure environment to a native MyST figure/subfigure block."""
    label_match = re.search(r"\\label\{([^{}]+)\}", body)
    caption_pos = body.find(r"\caption")
    caption = ""
    if caption_pos >= 0:
        brace = body.find("{", caption_pos + len(r"\caption"))
        if brace >= 0:
            raw_caption, _ = parse_braced(body, brace)
            caption = _clean_inline_tex(raw_caption)

    images = re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}",
        body,
    )
    if not images:
        return ""

    label = label_match.group(1).strip() if label_match else None
    if len(images) == 1:
        lines = [f":::{{figure}} {images[0]}"]
        if label:
            lines.append(f":label: {label}")
        lines.append(":align: center")
        if caption:
            lines.extend(["", caption])
        lines.append(":::")
        return "\n".join(lines)

    # Multiple images are intentionally left as direct children of the figure.
    # MyST interprets them as implicit subfigures while keeping the final text
    # as one figure-level caption spanning the full figure width.
    lines = [":::{figure}"]
    if label:
        lines.append(f":label: {label}")
    lines.append("")
    for image in images:
        lines.extend([f"![]({image})", ""])
    if caption:
        lines.append(caption)
    lines.append(":::")
    return "\n".join(lines)


def extract_figures(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Replace TeX figure environments with placeholders for native MyST figures."""
    structures: list[ExtractedStructure] = []
    pattern = re.compile(
        r"\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}",
        re.DOTALL,
    )

    def replace(match: re.Match[str]) -> str:
        markdown = figure_to_myst(match.group(1))
        if not markdown:
            return match.group(0)
        placeholder = f"BDLFIGUREPLACEHOLDER{len(structures):04d}"
        structures.append(ExtractedStructure(placeholder, markdown))
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
        if kind == "proof":
            # Use the explicit proof kind, but no custom title argument: MyST
            # supplies the semantic label "Proof" itself. A title argument would
            # be displayed parenthetically, e.g. "(Proof)".
            lines = [":::{prf:proof}", ":enumerated: false"]
        else:
            lines = [f":::{{prf:{kind}}}"]
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
    """Remove the source chapter heading before standalone MyST conversion.

    Individual chapter converters add one canonical page title and target after
    conversion. Keeping a synthetic chapter/section heading in the isolated TeX
    pass makes the web theme display the chapter title twice.
    """
    return re.sub(
        r"\\chapter(?:\[[^\]]*\])?\{[^{}]+\}\s*(?:\\label\{(?:chap|cha):[^{}]+\}\s*)?",
        "",
        text,
        count=1,
    )


def restore_extracted_structures(text: str, structures: list[ExtractedStructure]) -> str:
    """Replace placeholders in converted Markdown with native MyST structures."""
    for structure in structures:
        text = text.replace(structure.placeholder, structure.markdown)
    return restore_proof_directives(text)
