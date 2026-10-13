from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from chapter_conversion import (  # noqa: E402
    normalize_markdown_emphasis_spacing,
    prepare_latex,
)
from footnote_structures import extract_footnotes, restore_footnotes  # noqa: E402
from latex_normalize import (  # noqa: E402
    normalize_eqnarray,
    normalize_latex,
    normalize_notation,
    normalize_pre_extraction,
)
from myst_structures import algorithm_to_myst, figure_to_myst  # noqa: E402
from proof_structures import extract_proof_environments, proof_directive  # noqa: E402
from semantic_references import extract_references  # noqa: E402


def test_book_expectation_macro_is_expanded() -> None:
    source = r"\E[\theta\sim p(\theta)][p(y\mid x,\theta)]"
    converted = normalize_notation(source)
    assert r"\mathbb{E}_{\theta\sim p(\theta)}\left[p(y\mid x,\theta)\right]" == converted


def test_shared_gaussian_and_transpose_macros_are_expanded() -> None:
    source = r"\bm{r}\transpose \sim \N(\bm{0},\bm{I})"
    converted = normalize_notation(source)
    assert r"\bm{r}^{\top} \sim \mathcal{N}(\bm{0},\bm{I})" == converted


def test_chapter4_sampling_macros_are_expanded() -> None:
    source = r"\numPar, \idxPar, \sX"
    assert normalize_notation(source) == r"R, r, \mathsf{X}"


def test_adjacent_math_macro_keeps_control_word_boundary() -> None:
    assert normalize_notation(r"\mathrm{ESS}=\alpha\numMCsamples") == r"\mathrm{ESS}=\alpha M"


def test_variance_notation_is_normalized() -> None:
    assert normalize_notation(r"\mathbb{V}\textrm{ar}[X]") == r"\operatorname{Var}[X]"
    assert normalize_notation(r"\mathbb{V}\textrm{\emph{ar}}[X]") == r"\operatorname{Var}[X]"


def test_eqnarray_environment_names_become_align() -> None:
    source = r"""\begin{eqnarray}\nonumber
x & := & y,\\ \nonumber
z & = & w,\\
\label{eq:test}
u &\leq& v.
\end{eqnarray}"""
    converted = normalize_eqnarray(source)
    assert r"\begin{align}" in converted
    assert r"\end{align}" in converted
    assert r"x & := & y" in converted
    assert r"z & = & w" in converted
    assert r"u &\leq& v" in converted
    assert r"\label{eq:test}" in converted
    assert converted.count(r"\nonumber") == 2
    assert "eqnarray" not in converted


def test_roman_numerals_are_literal_text() -> None:
    assert normalize_notation(r"\romannumeral1) first; \romannumeral2) second") == (
        "i) first; ii) second"
    )


def test_kl_macro_supports_nested_braces() -> None:
    source = r"\KL{p(\theta\mid x)}{q_{\psi}(\theta\mid x)}"
    converted = normalize_notation(source)
    assert converted == (
        r"D_{KL}\left(p(\theta\mid x)\middle\|q_{\psi}(\theta\mid x)\right)"
    )
    assert r"\KL" not in converted


def test_capital_tilde_is_normalized() -> None:
    assert normalize_notation(r"\Tilde{w}_i") == r"\widetilde{w}_i"


def test_argmin_and_argmax_are_expanded() -> None:
    source = r"\argmin_{\psi} L(\psi), \argmax_{\theta} p(\theta)"
    converted = normalize_notation(source)
    assert r"\operatorname*{arg\,min}_{\psi}" in converted
    assert r"\operatorname*{arg\,max}_{\theta}" in converted


def test_nonsemantic_typography_is_normalized() -> None:
    source = r"\noindent{\bf Regression. } \textcolor{black}{Body} {\color{black} More}"
    converted = normalize_latex(source)
    assert converted == r"\textbf{Regression. } Body More"
    assert r"\noindent" not in converted
    assert r"\textcolor" not in converted
    assert r"\color" not in converted
    assert r"{\bf" not in converted


def test_markdown_emphasis_trailing_space_moves_outside_delimiters() -> None:
    source = "**Regression. ** If *important * text follows."
    converted = normalize_markdown_emphasis_spacing(source)
    assert converted == "**Regression.**  If *important*  text follows."


def test_adjacent_emphasis_spans_keep_their_spacing() -> None:
    for delimiter in ("*", "**"):
        source = (
            f"Popular choices include {delimiter}deterministic rounding{delimiter} "
            f"and {delimiter}stochastic rounding{delimiter}."
        )
        assert normalize_markdown_emphasis_spacing(source) == source


def test_bare_table_and_part_references_are_protected() -> None:
    source = r"Table \ref{tab:2}, (\ref{eq:1}), Book Part \ref{part:vi}"
    marked, structures = extract_references(source)
    for structure in structures:
        marked = marked.replace(structure.placeholder, structure.markdown)
    assert marked == "[](#tab:2), ([](#eq:1)), Book [](#part:vi)"


def test_abbreviated_semantic_references_are_normalized() -> None:
    source = r"""
\section{Sequential inference}
\label{sec:sequential}
See Eq.~\ref{eq:npe}, Sec.~\ref{sec:sequential}, and Fig.~\ref{fig:overview}.
"""
    converted, structures = extract_references(source)
    restored = converted
    for structure in structures:
        restored = restored.replace(structure.placeholder, structure.markdown)

    assert "Equation [](#eq:npe)" in restored
    assert "Section [Sequential inference](#sec:sequential)" in restored
    assert "Figure [](#fig:overview)" in restored
    assert r"Eq.~\ref" not in restored
    assert r"Sec.~\ref" not in restored
    assert r"Fig.~\ref" not in restored


def test_full_equation_word_preserves_lowercase() -> None:
    source = r"The algorithm satisfies the equation \eqref{eq:invariance}."
    converted, structures = extract_references(source)
    restored = converted
    for structure in structures:
        restored = restored.replace(structure.placeholder, structure.markdown)

    assert "satisfies the equation [](#eq:invariance)" in restored


def test_bare_eqref_does_not_invent_equation_word() -> None:
    source = r"The construction in \eqref{eq:DNN} is approximated at finite resolution."
    converted, structures = extract_references(source)
    restored = converted
    for structure in structures:
        restored = restored.replace(structure.placeholder, structure.markdown)

    assert "in [](#eq:DNN) is approximated" in restored
    assert "Equation [](#eq:DNN)" not in restored


def test_proof_references_let_myst_supply_object_prefix() -> None:
    source = r"See Algorithm \ref{alg:smc} and Theorem \ref{thm:main}."
    converted, structures = extract_references(source)
    restored = converted
    for structure in structures:
        restored = restored.replace(structure.placeholder, structure.markdown)

    assert "See [](#alg:smc) and [](#thm:main)." in restored
    assert "Algorithm [](#alg:smc)" not in restored
    assert "Theorem [](#thm:main)" not in restored
    assert r"\ref" not in restored


def test_simple_prose_group_around_reference_is_unwrapped() -> None:
    source = r"{Step 4 of Algorithm \ref{alg:mlsmc_dnn} applies exactly this kernel.}"
    converted, structures = extract_references(source)
    restored = converted
    for structure in structures:
        restored = restored.replace(structure.placeholder, structure.markdown)

    assert restored == "Step 4 of [](#alg:mlsmc_dnn) applies exactly this kernel."
    assert not restored.startswith("{")


def test_algorithm2e_aligned_tcp_comment_is_preserved() -> None:
    source = r"""
\caption{Example}
\label{alg:example}
\SetKwInOut{Input}{Inputs}
\SetKwInOut{Output}{Outputs}
\Input{dataset $\mathcal{D}$}
$x \gets 1$ \tcp*[r]{Compute value}
"""
    converted = algorithm_to_myst(source)
    assert "**Inputs:**" in converted
    assert "{math}`\\mathcal{D}`" in converted
    assert "*Note:* Compute value" in converted
    assert r"\tcp" not in converted


def test_algorithm2e_multiline_inline_math_is_one_step() -> None:
    source = r"""
\caption{SMC}
\label{alg:smc}
$Z_t^{\numMCsamples} \gets
  Z_{t-1}^{\numMCsamples} \cdot
  \frac{1}{\numMCsamples}\sum_{k=1}^{\numMCsamples}
  p(Y \mid \theta_{t-1}^k)^{\lambda_t-\lambda_{t-1}}$\;
"""
    converted = algorithm_to_myst(source)

    assert converted.count("1. ") == 1
    assert "{math}`Z_t^{M} \\gets Z_{t-1}^{M} \\cdot" in converted
    assert "\\frac{1}{M}\\sum_{k=1}^{M}" in converted
    assert "$Z_t" not in converted


def test_algorithm2e_repeat_loop_is_converted_recursively() -> None:
    source = r"""
\caption{Repeat example}
\label{alg:repeat}
\Repeat{termination criterion met}{
  \tcp{Effective sample size}
  $n \gets n+1$\;
}
"""
    converted = algorithm_to_myst(source)
    assert "**Repeat until** termination criterion met:" in converted
    assert "*Note:* Effective sample size" in converted
    assert r"\Repeat" not in converted
    assert r"\tcp" not in converted


def test_algorithm_caption_supports_nested_tex_braces() -> None:
    source = r"""
\caption{Sequential Monte Carlo to estimate $\mathbb{E}(\theta_n)$}
\label{alg:smc}
$x \gets 1$\;
"""
    converted = algorithm_to_myst(source)
    assert converted.startswith(
        r":::{prf:algorithm} Sequential Monte Carlo to estimate $\mathbb{E}(\theta_n)$"
    )
    assert r"\caption" not in converted


def test_figure_width_accepts_linewidth() -> None:
    source = r"""
\centering
\includegraphics[width=0.8\linewidth]{sampling_methods/sg_mcmc/fig/example.pdf}
\caption{Example figure.}
\label{fig:example}
"""
    converted = figure_to_myst(source)
    assert ":width: 80%" in converted


def test_figure_width_accepts_columnwidth_through_shared_prepass() -> None:
    source = r"""
\centering
\includegraphics[width=0.45\columnwidth]{sampling_methods/example/fig/panel.png}
\caption{Example figure.}
\label{fig:columnwidth}
"""
    converted = figure_to_myst(normalize_pre_extraction(source))
    assert ":width: 45%" in converted


def test_citations_are_preserved_inside_extracted_caption() -> None:
    source = r"""
\centering
\includegraphics{sampling_methods/sg_mcmc/fig/example.pdf}
\caption{Adapted from \cite{smith2020} and discussed by \citet{jones2021}.}
\label{fig:example}
"""
    converted = figure_to_myst(source)
    assert "[@smith2020]" in converted
    assert "@jones2021" in converted
    assert r"\cite" not in converted


def test_theorem_optional_title_and_citation_are_preserved() -> None:
    source = r"""
\begin{theorem}[Giles \citep{giles2008multilevel}]
\label{thm:VMLMC}
The estimator achieves the stated complexity.
\end{theorem}
"""
    marked, structures = extract_proof_environments(source)
    assert marked.strip() == "BDLPROOFPLACEHOLDER0000"
    assert len(structures) == 1
    structure = structures[0]
    assert structure.kind == "theorem"
    assert structure.label == "thm:VMLMC"
    assert structure.title == "Giles [@giles2008multilevel]"
    assert structure.body_tex == "The estimator achieves the stated complexity."

    restored = proof_directive(structure, structure.body_tex)
    assert restored.startswith(":::{prf:theorem} Giles [@giles2008multilevel]")
    assert ":label: thm:VMLMC" in restored
    assert "The estimator achieves the stated complexity." in restored


def test_commented_proof_environments_are_ignored_before_extraction() -> None:
    source = r"""
% \begin{proposition}
% \label{prop:commented}
% This must not become a proof structure.
% \end{proposition}

\begin{proposition}
\label{prop:active}
This one is active.
\end{proposition}
"""
    normalized, structures, proof_structures, footnote_structures = prepare_latex(source)

    assert len(proof_structures) == 1
    assert proof_structures[0].label == "prop:active"
    assert proof_structures[0].placeholder == "BDLPROOFPLACEHOLDER0000"
    assert "prop:commented" not in normalized
    assert all("prop:commented" not in structure.markdown for structure in structures)
    assert not footnote_structures


def test_footnotes_are_deterministic_and_not_duplicated() -> None:
    source = (
        r"Approximate posteriors can be overconfident"
        r"\footnote{This occurs when the approximation is too narrow.}, particularly."
    )
    marked, structures = extract_footnotes(source)
    assert marked == (
        "Approximate posteriors can be overconfident"
        "BDLFOOTNOTEPLACEHOLDER0001, particularly."
    )
    assert len(structures) == 1
    assert structures[0].identifier == "footnote-1"

    restored = restore_footnotes(marked, structures)
    assert restored.count("[^footnote-1]") == 2
    assert (
        "Approximate posteriors can be overconfident[^footnote-1], particularly."
        in restored
    )
    assert (
        "[^footnote-1]: This occurs when the approximation is too narrow."
        in restored
    )
    assert "BDLFOOTNOTEPLACEHOLDER" not in restored
