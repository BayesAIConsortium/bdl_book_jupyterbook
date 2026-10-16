from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from chapter_conversion import normalize_markdown_emphasis_spacing, prepare_latex  # noqa: E402
from content_structures import (  # noqa: E402
    CalloutStructure,
    callout_directive,
    extract_callouts,
    extract_subequations,
)
from latex_normalize import normalize_latex  # noqa: E402


def test_subequations_preserve_all_labels_and_repeat_continuation_lhs() -> None:
    source = r"""
\begin{subequations}
\begin{align}
p(\params\mid\traindata) &= Z^{-1}\exp(-\regrisk(\params)) \label{eq:energy} \\
&\approx Z^{-1}\exp(-q(\params)) \label{eq:taylor} \\
&= \N(\params;\params_*,C). \label{eq:gaussian}
\end{align}
\end{subequations}
"""
    marked, structures = extract_subequations(source)
    assert r"\begin{subequations}" not in marked
    assert len(structures) == 3
    for structure, label in zip(structures, ("eq:energy", "eq:taylor", "eq:gaussian"), strict=True):
        assert f":label: {label}" in structure.markdown
        assert r"p(\theta\mid\mathcal{D})" in structure.markdown
        assert r"\label" not in structure.markdown
        assert "&" not in structure.markdown
    assert r"\regrisk(\theta)" in structures[0].markdown  # Native project macro.
    assert r"\approx" in structures[1].markdown
    assert r"\mathcal{N}" in structures[2].markdown


def test_subequation_row_breaks_inside_matrices_are_preserved() -> None:
    source = r"""
\begin{subequations}\begin{align}
A &= \begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix} \label{eq:matrix} \\[1ex]
&= B \label{eq:b}
\end{align}\end{subequations}
"""
    _, structures = extract_subequations(source)
    assert len(structures) == 2
    assert r"1 & 2 \\ 3 & 4" in structures[0].markdown
    assert "A = B" in structures[1].markdown


def test_unsupported_subequation_structure_fails_instead_of_losing_labels() -> None:
    with pytest.raises(ValueError, match="Expected one align"):
        extract_subequations(r"\begin{subequations}\label{group}x=y\end{subequations}")


def test_callouts_preserve_lists_and_cross_references_for_body_conversion() -> None:
    source = r"""
\begin{callout}{Recipe}
\begin{enumerate}\item Train $\params_*$.\item See Equation~\ref{eq:energy}.\end{enumerate}
\end{callout}
"""
    normalized, structures, _, _ = prepare_latex(source)
    callout = next(s for s in structures if isinstance(s, CalloutStructure))
    assert r"\begin{callout}" not in normalized
    assert r"\begin{enumerate}" in callout.body_tex
    assert r"\theta_*" in callout.body_tex
    assert "BDLREFERENCEPLACEHOLDER" in callout.body_tex
    rendered = callout_directive(callout, "1. Train.\n2. Compute curvature.")
    assert rendered == ":::{admonition} Recipe\n\n1. Train.\n2. Compute curvature.\n:::"


def test_nested_callouts_get_distinct_tokens_and_valid_nested_fences() -> None:
    marked, callouts = extract_callouts(
        r"\begin{callout}{Outer}A \begin{callout}{Inner}B\end{callout} C\end{callout}"
    )
    assert len(callouts) == 2
    assert callouts[0].placeholder in callouts[1].body_tex
    assert callouts[1].placeholder in marked
    inner = callout_directive(callouts[0], "B")
    assert callout_directive(callouts[1], inner).startswith("::::{admonition} Outer")


def test_nested_italic_commands_are_flattened_before_markdown_export() -> None:
    assert normalize_latex(r"\emph{Need a \emph{Hessian} and its \textit{inverse}?}") == (
        r"\emph{Need a Hessian and its inverse?}"
    )


def test_math_stars_cannot_shift_adjacent_emphasis_delimiters() -> None:
    source = r"At $\theta_*$, captures the *local* geometry and *global* shape."
    assert normalize_markdown_emphasis_spacing(source) == source
    source = r"At {math}`\theta_*`, the *local* and *global* shape."
    assert normalize_markdown_emphasis_spacing(source) == source


def test_emphasis_cleanup_preserves_fenced_math_and_code_byte_for_byte() -> None:
    source = "```{math}\nx_* y_*\n```\n`a * b * c` and *important * text."
    assert normalize_markdown_emphasis_spacing(source) == (
        "```{math}\nx_* y_*\n```\n`a * b * c` and *important*  text."
    )


def test_optional_citation_notes_use_native_myst_affixes() -> None:
    from semantic_references import extract_references

    marked, structures = extract_references(r"See \citep[e.g.][]{one,two}.")
    for structure in structures:
        marked = marked.replace(structure.placeholder, structure.markdown)
    assert marked == "See {cite:p}`{e.g.}one,two`."

    _, structures = extract_references(r"\citep[p. 5]{one}")
    assert structures[0].markdown == "{cite:p}`one{p. 5}`"
