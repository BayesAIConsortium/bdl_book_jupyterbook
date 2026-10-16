from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import chapter_conversion  # noqa: E402
from chapter_conversion import ChapterConfig, convert_chapter  # noqa: E402
from tex_metadata import source_chapter_title  # noqa: E402


def test_full_title_takes_precedence_over_optional_short_title() -> None:
    assert source_chapter_title(
        r"\chapter[Introduction]{Introduction to Laplace approximations}"
    ) == "Introduction to Laplace approximations"


def test_title_parser_handles_formatting_comments_and_control_word_boundaries() -> None:
    source = "% \\chapter{Commented title}\n" + (
        r"\chapterinitial{A}{stute} \chapter{An \textbf{updated} title \& examples}"
    )
    assert source_chapter_title(source) == "An updated title & examples"
    assert source_chapter_title(r"\chapterinitial{A}{stute} readers") is None


@pytest.mark.parametrize(
    ("source", "label", "fallback", "expected"),
    [
        (
            r"\chapter[Introduction]{Introduction to Laplace approximations}",
            "cha:laplace:introduction", "Introduction", "Introduction to Laplace approximations",
        ),
        (
            r"\chapter{Stochastic Gradient MCMC}",
            "chap:sampling_methods_sg_mcmc", "Stale title", "Stochastic gradient MCMC",
        ),
        ("Body without a chapter command.", "chap:example", "Fallback title", "Fallback title"),
    ],
)
def test_conversion_uses_source_title_and_preserves_editorial_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    source: str, label: str, fallback: str, expected: str,
) -> None:
    input_path = tmp_path / "main.tex"
    output_path = tmp_path / "chapter.md"
    input_path.write_text(source + "\nBody.")
    monkeypatch.setattr(chapter_conversion, "run_myst_with_structures", lambda *args: "Body.")
    config = ChapterConfig(fallback, label, input_path, output_path, "example")
    convert_chapter(config)
    assert output_path.read_text() == f"({label})=\n# {expected}\n\nBody."
