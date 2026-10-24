from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from latex_normalize import normalize_expectation  # noqa: E402
from myst_structures import figure_to_myst  # noqa: E402


def test_nested_expectations_keep_inner_condition_and_both_bodies() -> None:
    source = r"\E[D\sim\mathrm{split}(D)][\log \E[\theta\sim p(\theta\mid D)][p(y\mid\theta)]]"
    assert normalize_expectation(source) == (
        r"\mathbb{E}_{D\sim\mathrm{split}(D)}\left[\log "
        r"\mathbb{E}_{\theta\sim p(\theta\mid D)}\left[p(y\mid\theta)\right]\right]"
    )


def test_expectation_arguments_respect_braced_and_escaped_delimiters() -> None:
    assert normalize_expectation(r"\E[x][f({[a,b]},\E[z][g(z)])]") == (
        r"\mathbb{E}_{x}\left[f({[a,b]},\mathbb{E}_{z}\left[g(z)\right])\right]"
    )
    assert normalize_expectation(r"\E[x][\left\lVert x\right\rVert]") == (
        r"\mathbb{E}_{x}\left[\left\lVert x\right\rVert\right]"
    )


def test_simple_expectations_retain_existing_output() -> None:
    source = r"\E[x][f(x)] + \E[g(x)] + \E + \Euler"
    assert normalize_expectation(source) == (
        r"\mathbb{E}_{x}\left[f(x)\right] + \mathbb{E}\left[g(x)\right]"
        r" + \mathbb{E} + \Euler"
    )


def test_unclosed_expectation_fails_instead_of_losing_math() -> None:
    with pytest.raises(ValueError, match="Unclosed optional argument"):
        normalize_expectation(r"\E[x][f(x)")


def test_multi_image_width_is_opt_in_and_does_not_override_source_widths() -> None:
    source = r"""
\includegraphics{first.pdf}\hfill\includegraphics{second.pdf}
\caption{Pair.}\label{fig:pair}
"""
    original = figure_to_myst(source)
    assert "![](first.pdf)" in original and ":width:" not in original
    converted = figure_to_myst(source, default_multi_image_width=45)
    assert converted.count(":width: 45%") == 2
    assert "grid" not in converted and ":label: fig:pair" in converted
    explicit = figure_to_myst(
        source.replace(r"\includegraphics", r"\includegraphics[width=.3\textwidth]"),
        default_multi_image_width=45,
    )
    assert explicit.count(":width: 30%") == 2


def test_fallback_does_not_change_single_figure() -> None:
    source = r"\includegraphics{plot.png}\caption{Plot.}"
    assert figure_to_myst(source, default_multi_image_width=45) == figure_to_myst(source)
