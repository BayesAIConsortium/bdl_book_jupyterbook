from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from chapter_conversion import _is_block_structure, prepare_latex  # noqa: E402
from footnote_structures import extract_footnotes  # noqa: E402
from myst_structures import figure_to_myst  # noqa: E402
from semantic_references import extract_references  # noqa: E402
from table_structures import extract_booktabs_tables  # noqa: E402


def test_subfigures_keep_parent_caption_and_every_panel_label() -> None:
    source = r"""
\begin{subfigure}[t]{\dimexpr(\textwidth-\columnsep)/2\relax}
\includegraphics{first.pdf}\caption{}\label{fig:first}
\end{subfigure}\hfill
\begin{subfigure}[t]{.5\textwidth}
\includegraphics{second.pdf}\caption{Second panel}\label{fig:second}
\end{subfigure}
\caption{Shared caption with $\params$ and \citet{paper}.}\label{fig:both}
"""
    markdown = figure_to_myst(source)
    assert ":label: fig:both" in markdown
    assert "(fig:first)=\n```{image} first.pdf\n:width: 45%" in markdown
    assert "(fig:second)=\n```{image} second.pdf\n:width: 45%\n:alt: Second panel" in markdown
    assert "Shared caption with $\\theta$ and @paper." in markdown
    assert ":class:" not in markdown
    assert markdown.count(":width: 45%") == 2
    assert r"\dimexpr" not in markdown


def test_plain_figure_retains_existing_conversion() -> None:
    assert figure_to_myst(
        r"\includegraphics[width=.7\textwidth]{plot.png}\caption{Plot}\label{fig:plot}"
    ) == (
        ":::{figure} plot.png\n:label: fig:plot\n:align: center\n"
        ":width: 70%\n\nPlot\n:::"
    )


def test_narrative_citation_keeps_optional_note_and_style() -> None:
    marked, structures = extract_references(r"See \citet[Fig.~3]{immer2021improving}.")
    for structure in structures:
        marked = marked.replace(structure.placeholder, structure.markdown)
    assert marked == "See @immer2021improving [Fig. 3]."


def test_footnote_equation_references_are_native_links() -> None:
    _, footnotes = extract_footnotes(
        r"Text\footnote{From Equation~\ref{eq:taylor} to \eqref{eq:gaussian} at $\params_*$.}"
    )
    assert footnotes[0].body == (
        r"From Equation [](#eq:taylor) to [](#eq:gaussian) at $\theta_*$."
    )


def test_booktabs_table_preserves_header_cells_caption_and_label() -> None:
    source = r"""
\begin{table}[t]
\begin{tabularx}{\textwidth}{@{}l*{2}{>{\raggedright\arraybackslash}X}@{}}
\toprule
Approx. & Storage & Correlations \\
\midrule
Diag. & $\mathcal O(\dimparams)$ & None \\
KFAC & Factors & Within-layer \\
\bottomrule
\end{tabularx}
\caption{Curvature comparison.}\label{tab:curvature}
\end{table}
"""
    _, structures = extract_booktabs_tables(source)
    assert len(structures) == 1
    assert _is_block_structure(structures[0])
    assert structures[0].markdown == (
        ":::{table} Curvature comparison.\n:label: tab:curvature\n\n"
        "| Approx. | Storage | Correlations |\n| --- | --- | --- |\n"
        "| Diag. | $\\mathcal O(P)$ | None |\n"
        "| KFAC | Factors | Within-layer |\n:::"
    )
    _, untouched = extract_booktabs_tables(
        r"\begin{table}\begin{tabular}{cc}a&b\end{tabular}\end{table}"
    )
    assert untouched == []


def test_linearized_chapter_structural_labels_survive_extraction() -> None:
    # The source chapter is available locally during migration; skip this
    # integration check on machines without the authoritative TeX checkout.
    from common import tex_path

    path = tex_path("laplace_approximations", "linearized", "main.tex")
    if not path.exists():
        import pytest
        pytest.skip("Authoritative TeX checkout unavailable")
    source = path.read_text()
    normalized, structures, proofs, footnotes = prepare_latex(source)
    material = normalized + "\n" + "\n".join(s.markdown for s in structures)
    material += "\n" + "\n".join(p.body_tex for p in proofs)
    material += "\n" + "\n".join(p.label or "" for p in proofs)
    for label in re.findall(r"\\label\{([^{}]+)\}", source):
        if label == "cha:laplace:linearized":  # Added by the shared driver.
            continue
        assert label in material, label
    assert all(r"\ref{" not in note.body for note in footnotes)


def test_dataset_subscript_stays_in_math_mode() -> None:
    from latex_normalize import normalize_notation

    assert normalize_notation(r"p_\text{\traindata}(\inputs)") == (
        r"p_{\mathcal{D}}(x)"
    )
