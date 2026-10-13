from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from latex_normalize import normalize_latex, normalize_notation  # noqa: E402
from myst_structures import algorithm_to_myst, extract_figures, extract_headings  # noqa: E402
from table_structures import extract_metric_tables  # noqa: E402


def test_low_precision_math_macros_are_expanded() -> None:
    source = r"\rd t, \norm{\theta-\theta'}_2, \Var_v^{hmc}"
    converted = normalize_notation(source)

    assert r"\,\mathrm{d} t" in converted
    assert r"\left\lVert \theta-\theta' \right\rVert_2" in converted
    assert r"\mathbb{V}_v^{hmc}" in converted
    assert r"\rd" not in converted
    assert r"\norm" not in converted
    assert r"\Var" not in converted


def test_legacy_it_declaration_is_normalized_without_touching_item() -> None:
    source = r"{\it dissipative} \paragraph{\it Sampling Thinning} \item value"
    converted = normalize_latex(source)

    assert r"\textit{dissipative}" in converted
    assert r"\paragraph{\textit{Sampling Thinning}}" in converted
    assert r"\item value" in converted
    assert r"{\it" not in converted


def test_scriptsize_is_dropped_as_nonsemantic_typography() -> None:
    source = r"Method & {\scriptsize$1.23\pm0.04$} & {\scriptsize 98.7}"
    converted = normalize_latex(source)

    assert r"\scriptsize" not in converted
    assert r"$1.23\pm0.04$" in converted
    assert "98.7" in converted


def test_algorithm2e_foreach_else_and_kwret_are_converted() -> None:
    source = r"""
\caption{Variance-corrected quantization}
\label{alg:vc}
\ForEach{$i$}{
  \If{$v > v_s$}{
    $x \gets 1$\;
  }
  \Else{
    $x \gets 0$\;
  }
}
\KwRet{$x$}
"""
    converted = algorithm_to_myst(source)

    assert "**For each** {math}`i`:" in converted
    assert "**If** {math}`v > v_s`:" in converted
    assert "**Else:**" in converted
    assert "**Return** {math}`x`" in converted
    assert r"\ForEach" not in converted
    assert r"\Else" not in converted
    assert r"\KwRet" not in converted


def test_algorithm_text_formatting_is_converted() -> None:
    converted = algorithm_to_myst(r"\textbf{sample} $c_i$\; \textbf{clip} $x$\;")
    assert "**sample** {math}`c_i`" in converted
    assert "**clip** {math}`x`" in converted
    assert r"\textbf" not in converted


def test_heading_levels_and_labels_survive_fragment_boundaries() -> None:
    source = r"\subsection{SGHMC}\label{sec:hmc} Body \paragraph{\textbf{Experiments}}"
    marked, structures = extract_headings(source)
    assert r"\subsection" not in marked
    assert structures[0].markdown == "(sec:hmc)=\n### SGHMC"
    assert structures[1].markdown == "#### **Experiments**"


def test_metric_table_headers_rows_and_standard_errors_are_preserved() -> None:
    source = r"""
\begin{table}
\begin{tabular}{l|cc|cc}
\hline
 &\multicolumn{2}{c|}{Test Errors (\%)} & \multicolumn{2}{c}{ECE $\downarrow$(\%)}\\
 &CIFAR-10 &CIFAR-100 &CIFAR-10 &CIFAR-100\\
 8-bit Fixed Point &&\\\hline
 VC SGLDLP-L &7.13 {\scriptsize$\pm$0.01} &26.62 {\scriptsize$\pm$0.16} &0.60 &3.19\\
\end{tabular}
\caption{Test Error and ECE.}\label{ch5:tab:1}
\end{table}
"""
    marked, structures = extract_metric_tables(source)
    assert marked.strip() == "BDLTABLEPLACEHOLDER0000"
    table = structures[0].markdown
    assert ":label: ch5:tab:1" in table
    assert "Test Errors (%) — CIFAR-100" in table
    assert "ECE $\\downarrow$(%) — CIFAR-10" in table
    assert "| VC SGLDLP-L | 7.13 ± 0.01 | 26.62 ± 0.16 | 0.60 | 3.19 |" in table
    assert "| **8-bit Fixed Point** |  |  |  |  |" in table
    assert all(line.count("|") == 6 for line in table.splitlines() if line.startswith("|"))


def test_starred_multi_image_figure_is_extracted() -> None:
    source = r"""
\begin{figure*}
\centering
\includegraphics[width=4cm]{sampling_methods/example/fig/a.pdf}
\includegraphics[width=4cm]{sampling_methods/example/fig/b.pdf}
\caption{Two panels with na\"ive sampling.}
\label{fig:star}
\end{figure*}
"""
    marked, structures = extract_figures(source)

    assert marked.strip() == "BDLFIGUREPLACEHOLDER0000"
    assert len(structures) == 1
    markdown = structures[0].markdown
    assert "sampling_methods/example/fig/a.pdf" in markdown
    assert "sampling_methods/example/fig/b.pdf" in markdown
    assert ":label: fig:star" in markdown
    assert "Two panels with naïve sampling." in markdown
