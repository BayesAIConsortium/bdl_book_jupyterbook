from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from myst_structures import algorithm_to_myst  # noqa: E402
from semantic_references import extract_references  # noqa: E402


def test_adaptation_branches_preserve_conditions_and_nested_assignments() -> None:
    source = r"""
\uIf{$\max(d_0,d_+,d_-)=d_+$}{ $\phi'\gets\phi_+$\; }
\uElseIf{$\max(d_0,d_+,d_-)=d_-$}{ $\phi'\gets\phi_-$\; }
\Else{ $\phi'\gets\phi_0$\; }
\KwRet{$\phi'$}\;
"""
    converted = algorithm_to_myst(source)
    assert "1. **If** {math}`\\max(d_0,d_+,d_-)=d_+`:" in converted
    assert "1. **Else if** {math}`\\max(d_0,d_+,d_-)=d_-`:" in converted
    assert "1. **Else:**" in converted
    for value in ("+", "-", "0"):
        assert f"    1. {{math}}`\\phi'\\gets\\phi_{value}`" in converted
    assert r"\uIf" not in converted
    assert r"\uElseIf" not in converted
    assert "1. }" not in converted


def test_cross_chapter_section_reference_uses_native_title_substitution() -> None:
    source = r"See Section~\ref{sec:mh} and Sec.\ref{sec:mala}."
    marked, structures = extract_references(source)
    for structure in structures:
        marked = marked.replace(structure.placeholder, structure.markdown)
    assert marked == "See [Section {name}](#sec:mh) and [Section {name}](#sec:mala)."
    assert "link" not in marked


def test_figure_reference_uses_native_prefix_without_duplication() -> None:
    marked, structures = extract_references(r"Figure~\ref{fig:discrete} shows the proposal.")
    for structure in structures:
        marked = marked.replace(structure.placeholder, structure.markdown)
    assert marked == "[](#fig:discrete) shows the proposal."
