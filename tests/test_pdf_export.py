from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pdf_export import (  # noqa: E402
    export_markdown,
    normalize_math_paragraphs,
    repair_figures,
    repair_references,
)


def test_pdf_only_algorithm_container_keeps_body_and_label() -> None:
    source = (
        "+++\n:::{prf:algorithm} Metropolis\n:label: alg:mh\n\n"
        "1. Draw $x$.\n2. Accept or reject.\n:::\n"
    )
    result = export_markdown(source)
    assert "{prf:axiom} Metropolis" in result
    assert ":label: alg:mh\n\n1. Draw $x$.\n2. Accept or reject." in result
    assert "+++" not in result
    with pytest.raises(ValueError):
        export_markdown(":::{proof:axiom}\nBody\n:::")


def test_native_nested_floats_become_subfigures_with_parent_label() -> None:
    md = ":::{figure}\n:label: fig:parent\n\nImages\n\nCaption.\n:::\n"
    child = (
        "\\begin{figure}[!htbp]\n\\centering\n"
        "\\includegraphics[width=0.45\\linewidth]{panel.png}\n"
        "\\caption[]{}\n\\label{fig:panel-a}\n\\end{figure}\n"
    )
    tex = (
        "Before\n\\begin{figure}[!htbp]\n\\centering\n" + child + "\n" + child
        + "\\caption[]{Caption.}\n\\label{fig:panel-a}\n\\end{figure}\nAfter"
    )
    result = repair_figures(tex, md)
    assert result.count(r"\begin{figure}") == 1
    assert result.count(r"\begin{subfigure}[t]{0.45\linewidth}") == 2
    assert result.count(r"\includegraphics[width=\linewidth]{panel.png}") == 2
    assert r"\label{fig:parent}" in result
    assert result.count(r"\label{fig:panel-a}") == 2
    assert r"\end{subfigure}\hfill\begin{subfigure}" in result
    assert result.startswith("Before\n") and result.endswith("\nAfter")


def test_unmatched_figure_structure_fails_instead_of_misassigning_labels() -> None:
    with pytest.raises(ValueError):
        repair_figures(r"\begin{figure}Image\end{figure}", "# No figures")


def test_math_blank_lines_do_not_change_prose_or_equations() -> None:
    tex = "Text.\n\n" + "\\begin{align}\n\nx &= y\n\n\\end{align}\n\nAfter."
    result = normalize_math_paragraphs(tex)
    assert result == "Text.\n\n\\begin{align}\nx &= y\n\\end{align}\n\nAfter."


def test_labelled_align_retains_its_directive_label_and_single_number() -> None:
    source = (
        "```{math}\n:label: eq:test\n\n\\begin{align}\n"
        "x &= y \\notag\\\\\nz &= w\n\\end{align}\n```\n"
    )
    result = export_markdown(source)
    assert ":label: eq:test" in result
    assert r"\begin{aligned}" in result and r"\end{aligned}" in result
    assert r"\notag" not in result


def test_standalone_pdf_links_remote_equations_and_normalizes_local_label_case() -> None:
    tex = r"\label{alg:sis} Axiom~\ref{alg:SIS}; Equation (\ref{eq:other})."
    references = {"eq:other": {
        "number": "22", "url": "/laplace-approximations/linearized", "html_id": "eq-other",
    }}
    result = repair_references(tex, references)
    assert r"Algorithm~\ref{alg:sis}" in result
    assert r"\href{https://bdlbook.org/laplace-approximations/linearized\#eq-other}{22}" in result
    with pytest.raises(ValueError):
        repair_references(r"\ref{missing}", {})

