from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from latex_normalize import normalize_notation  # noqa: E402
from myst_structures import algorithm_to_myst, figure_to_myst  # noqa: E402


def test_book_expectation_macro_is_expanded() -> None:
    source = r"\E[\theta\sim p(\theta)][p(y\mid x,\theta)]"
    converted = normalize_notation(source)
    assert r"\mathbb{E}_{\theta\sim p(\theta)}\left[p(y\mid x,\theta)\right]" == converted


def test_shared_gaussian_and_transpose_macros_are_expanded() -> None:
    source = r"\bm{r}\transpose \sim \N(\bm{0},\bm{I})"
    converted = normalize_notation(source)
    assert r"\bm{r}^{\top} \sim \mathcal{N}(\bm{0},\bm{I})" == converted


def test_roman_numerals_are_literal_text() -> None:
    assert normalize_notation(r"\romannumeral1) first; \romannumeral2) second") == (
        "i) first; ii) second"
    )


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
    assert "*Note:* Compute value" in converted
    assert r"\tcp" not in converted


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


def test_figure_width_accepts_linewidth() -> None:
    source = r"""
\centering
\includegraphics[width=0.8\linewidth]{sampling_methods/sg_mcmc/fig/example.pdf}
\caption{Example figure.}
\label{fig:example}
"""
    converted = figure_to_myst(source)
    assert ":width: 80%" in converted


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
