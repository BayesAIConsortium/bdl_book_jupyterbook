from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chapter_conversion import prepare_latex  # noqa: E402
from latex_normalize import normalize_text_control_spaces  # noqa: E402


@pytest.mark.parametrize("protected", [
    r"$x\ y$", r"$$x\ y$$", r"\(x\ y\)", r"\[x\ y\]",
    r"\begin{align}x\ y\end{align}",
    r"\begin{equation*}x\ y\end{equation*}",
    r"\begin{subequations}\begin{align}x\ y\end{align}\end{subequations}",
    r"\verb|x\ y|", r"\verb*+x\ y+",
    r"\texttt{x\ {nested\ code}}", r"`x\ y`", "```\nx\\ y\n```",
    r"\begin{verbatim}x\ y\end{verbatim}",
    r"\begin{lstlisting}x\ y\end{lstlisting}",
    r"\begin{minted}{python}x\ y\end{minted}",
    r"x\\ y", r"% comment\ space",
])
def test_control_spaces_protect_math_and_code(protected: str) -> None:
    source = "before\\ text " + protected + "\nafter\\ text"
    expected = "before text " + protected + "\nafter text"
    assert normalize_text_control_spaces(source) == expected


def test_only_control_spaces_are_normalized() -> None:
    source = r"vs.\ online; i.i.d.\ draws; \quad \! \; \~x; cost \$5"
    expected = r"vs. online; i.i.d. draws; \quad \! \; \~x; cost \$5"
    assert normalize_text_control_spaces(source) == expected
    assert normalize_text_control_spaces(expected) == expected


def test_control_spaces_are_normalized_before_heading_extraction() -> None:
    _, structures, _, _ = prepare_latex(
        r"\subsection{Post-hoc vs.\ online model selection}"
    )
    assert structures[0].markdown == "### Post-hoc vs. online model selection"
