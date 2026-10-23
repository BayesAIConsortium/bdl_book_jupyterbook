#!/usr/bin/env python3
"""Reusable conversion of theorem-like TeX environments to native MyST proofs."""

from __future__ import annotations

import re
from dataclasses import dataclass

from latex_normalize import normalize_notation
from myst_structures import parse_braced

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
class ProofStructure:
    """One theorem-like environment protected from the main MyST pass."""

    placeholder: str
    kind: str
    label: str | None
    title: str | None
    body_tex: str


def _clean_title(text: str) -> str:
    """Convert conservative inline TeX used in theorem titles to MyST Markdown."""
    text = text.strip()
    # Optional theorem titles may be wrapped in a TeX scoping group. Remove
    # only a balanced group around the entire title, keeping inner math intact.
    while text.startswith("{"):
        inner, end = parse_braced(text, 0)
        if end != len(text):
            break
        text = inner.strip()
    text = normalize_notation(text)
    text = re.sub(
        r"\\(?:cite|citep)\{([^{}]+)\}",
        lambda match: "[" + "; ".join(
            f"@{key.strip()}" for key in match.group(1).split(",") if key.strip()
        ) + "]",
        text,
    )
    text = re.sub(
        r"\\citet\{([^{}]+)\}",
        lambda match: (
            f"@{match.group(1).strip()}"
            if "," not in match.group(1)
            else "[" + "; ".join(
                f"@{key.strip()}" for key in match.group(1).split(",") if key.strip()
            ) + "]"
        ),
        text,
    )
    text = re.sub(r"\\(?:emph|textit)\{([^{}]*)\}", r"*\1*", text)
    text = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", text)
    text = text.replace("~", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract_proof_environments(text: str) -> tuple[str, list[ProofStructure]]:
    """Replace theorem-like environments with one opaque placeholder each.

    The theorem body is converted separately from the main chapter. This avoids
    relying on a pair of boundary tokens surviving arbitrary MyST block parsing.
    """
    structures: list[ProofStructure] = []

    for kind in PROOF_KINDS:
        pattern = re.compile(
            rf"\\begin\{{{kind}\}}(?:\[([^\]]*)\])?(.*?)\\end\{{{kind}\}}",
            re.DOTALL,
        )

        def replace(match: re.Match[str], proof_kind: str = kind) -> str:
            title = _clean_title(match.group(1)) if match.group(1) else None
            body = match.group(2)
            label_match = re.match(r"\s*\\label\{([^{}]+)\}\s*", body)
            label = label_match.group(1) if label_match else None
            if label_match:
                body = body[label_match.end() :]

            placeholder = f"BDLPROOFPLACEHOLDER{len(structures):04d}"
            structures.append(
                ProofStructure(
                    placeholder=placeholder,
                    kind=proof_kind,
                    label=label,
                    title=title,
                    body_tex=body.strip(),
                )
            )
            return f"\n\n{placeholder}\n\n"

        text = pattern.sub(replace, text)

    return text, structures


def proof_directive(structure: ProofStructure, body_markdown: str) -> str:
    """Build one native MyST proof directive from converted body Markdown."""
    if structure.kind == "proof":
        lines = [":::{prf:proof}", ":enumerated: false"]
    else:
        heading = f":::{{prf:{structure.kind}}}"
        if structure.title:
            heading += f" {structure.title}"
        lines = [heading]

    if structure.label:
        lines.append(f":label: {structure.label}")
    lines.extend(["", body_markdown.strip(), ":::"])
    return "\n".join(lines)
