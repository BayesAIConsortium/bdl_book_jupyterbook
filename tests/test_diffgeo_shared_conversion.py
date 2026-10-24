from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from myst_structures import figure_to_myst  # noqa: E402
from proof_structures import extract_proof_environments  # noqa: E402


def test_adjacent_equal_width_images_retain_original_width_directives() -> None:
    source = r"""
\includegraphics[width=.45\textwidth,trim={3cm 3cm 0 0}]{left.pdf}
\includegraphics[width=.45\textwidth,trim={3cm 3cm 0 0}]{right.pdf}
\caption{Left and right; see Equation~\ref{eq:fibers}.}\label{fig:manifolds}
"""
    markdown = figure_to_myst(source)
    assert ":class:" not in markdown
    assert ":label: fig:manifolds" in markdown
    assert "```{image} left.pdf" in markdown and "```{image} right.pdf" in markdown
    assert markdown.count(":width: 45%") == 2
    assert "Equation [](#eq:fibers)" in markdown
    assert r"\ref" not in markdown


def test_explicit_line_break_keeps_stacked_figures() -> None:
    source = r"""
\includegraphics[width=.45\textwidth]{left.pdf}\\
\includegraphics[width=.45\textwidth]{right.pdf}
\caption{Stacked plots.}
"""
    markdown = figure_to_myst(source)
    assert ":class:" not in markdown
    assert markdown.count(":width: 45%") == 2


def test_full_width_images_do_not_gain_two_column_layout() -> None:
    source = r"""
\includegraphics[width=\textwidth]{top.pdf}
\includegraphics[width=\textwidth]{bottom.pdf}
\caption{Stacked plots.}
"""
    assert ":class:" not in figure_to_myst(source)


def test_subfigure_caption_references_use_the_same_conversion() -> None:
    source = r"""
\begin{subfigure}{.5\textwidth}
\includegraphics{left.pdf}\caption{See Equation~\ref{eq:left}.}\label{fig:left}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
\includegraphics{right.pdf}\caption{}\label{fig:right}
\end{subfigure}
\caption{See Section~\ref{sec:geometry}.}\label{fig:both}
"""
    markdown = figure_to_myst(source)
    assert "Equation [](#eq:left)" in markdown
    assert "[Section {name}](#sec:geometry)" in markdown
    assert "(fig:left)=" in markdown and "(fig:right)=" in markdown
    assert r"\ref" not in markdown


def test_grouped_theorem_title_keeps_citation_and_inner_math_braces() -> None:
    source = r"""
\begin{theorem}[{Proposition 5.6 in \citet{ay2017information}}]
Statement.
\end{theorem}
"""
    _, proofs = extract_proof_environments(source)
    assert proofs[0].title == "Proposition 5.6 in @ay2017information"
    _, proofs = extract_proof_environments(
        r"\begin{theorem}[{Case $\mathcal{M}$}]Statement.\end{theorem}"
    )
    assert proofs[0].title == r"Case $\mathcal{M}$"
