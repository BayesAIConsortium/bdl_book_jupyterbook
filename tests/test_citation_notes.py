from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from semantic_references import extract_references  # noqa: E402
from text_normalize import normalize_tex_citation_note  # noqa: E402


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (r"See \citep[Theorem~1]{sell2023trace}.",
         "See {cite:p}`sell2023trace{Theorem 1}`."),
        (r"\citep[see~e.g.][]{one,two}", "{cite:p}`{see e.g.}one,two`"),
        (r"\citep[SGHMC,][]{chen2014stochastic}", "[SGHMC, @chen2014stochastic]"),
        (r"\citep[e.g.,][]{one,two}", "[e.g., @one; @two]"),
        (r"\citep[see~e.g.,][Theorem~1]{one,two}", "[see e.g., @one; @two, Theorem 1]"),
    ],
)
def test_citation_notes_preserve_native_syntax_and_spacing(source: str, expected: str) -> None:
    marked, structures = extract_references(source)
    for structure in structures:
        marked = marked.replace(structure.placeholder, structure.markdown)
    assert marked == expected


def test_citation_note_spacing_and_accents_are_normalized_as_prose() -> None:
    assert normalize_tex_citation_note(r"Th\'eor\`eme~1,\,p.\ 5") == "Théorème 1, p. 5"


def test_note_cleanup_does_not_modify_math_elsewhere_in_the_source() -> None:
    source = r"$\tilde{x}\sim\pi$; see \citep[Theorem~1]{one}."
    marked, _ = extract_references(source)
    assert r"$\tilde{x}\sim\pi$" in marked
