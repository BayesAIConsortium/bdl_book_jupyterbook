#!/usr/bin/env python3
"""Shared normalization for book-specific LaTeX before MyST conversion.

This module handles reusable notation expansion and non-semantic typographic
cleanup. Semantic structures such as algorithms, theorems, proofs, and chapter
headings belong in myst_structures.py.
"""

from __future__ import annotations

import re
from collections.abc import Callable


COMPOUND_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\network(\inputs,\params)", r"f(x,\theta)"),
)

SIMPLE_MATH_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\traininputs", "X"),
    (r"\traintargets", "Y"),
    (r"\traindata", r"\mathcal{D}"),
    (r"\testinput", r"x_*"),
    (r"\testoutput", r"y_*"),
    (r"\inputspace", r"\mathcal{X}"),
    (r"\targetspace", r"\mathcal{Y}"),
    (r"\paramspace", r"\Theta"),
    (r"\diminputs", "D"),
    (r"\dimoutputs", "C"),
    (r"\dimparams", "P"),
    (r"\numtraindata", "N"),
    (r"\numMCsamples", "M"),
    (r"\probability", "p"),
    (r"\approxprob", "q"),
    (r"\targetprob", r"\pi"),
    (r"\testfunction", "h"),
    (r"\unnormalized", r"\kappa"),
    (r"\mcmckernelB", r"\mathcal{K}"),
    (r"\mcmckernel", r"\mathcal{M}"),
    (r"\partitionfunc", "Z"),
    (r"\network", "f"),
    (r"\activation", r"\sigma"),
    (r"\softmax", "S"),
    (r"\netdepth", "L"),
    (r"\loss", r"\ell"),
    (r"\risk", r"\mathcal{L}"),
    (r"\regularizer", r"\Omega"),
    (r"\Reals", r"\mathbb{R}"),
    (r"\Nats", r"\mathbb{N}"),
    (r"\N", r"\mathcal{N}"),
    (r"\transpose", r"^{\top}"),
    (r"\argmin", r"\operatorname*{arg\,min}"),
    (r"\argmax", r"\operatorname*{arg\,max}"),
    (r"\params", r"\theta"),
    (r"\inputs", "x"),
    (r"\targets", "y"),
    (r"\weights", "A"),
    (r"\biases", "b"),
    (r"\constant", "K"),
)


TYPOGRAPHIC_COMMANDS: tuple[str, ...] = (
    r"\smallskip",
    r"\medskip",
    r"\bigskip",
)


def _parse_braced(text: str, start: int) -> tuple[str, int]:
    """Parse one balanced braced argument and return its contents and next index."""
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        raise ValueError(f"Expected '{{' at position {start}")

    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{" and (pos == 0 or text[pos - 1] != "\\"):
            depth += 1
        elif text[pos] == "}" and (pos == 0 or text[pos - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos], pos + 1
    raise ValueError("Unclosed brace group")


def _replace_braced_command(
    text: str,
    command: str,
    nargs: int,
    render: Callable[[list[str]], str],
) -> str:
    """Replace a command with balanced braced arguments throughout a string."""
    pieces: list[str] = []
    pos = 0
    while True:
        command_pos = text.find(command, pos)
        if command_pos < 0:
            pieces.append(text[pos:])
            break

        # Avoid treating a prefix of a longer control word as the requested command.
        after_command = command_pos + len(command)
        if after_command < len(text) and text[after_command].isalpha():
            pieces.append(text[pos:after_command])
            pos = after_command
            continue

        pieces.append(text[pos:command_pos])
        cursor = after_command
        arguments: list[str] = []
        try:
            for _ in range(nargs):
                argument, cursor = _parse_braced(text, cursor)
                arguments.append(argument)
        except ValueError:
            pieces.append(text[command_pos:after_command])
            pos = after_command
            continue

        pieces.append(render(arguments))
        pos = cursor

    return "".join(pieces)


def normalize_indicator(text: str) -> str:
    r"""Expand the book's indicator-function macro to standard LaTeX.

    The authoritative TeX macro is ``\Ind[o] = \mathds 1(o)``. MyST/KaTeX does
    not know that book-specific command, so retain the same round-bracket
    semantics using a standard blackboard-bold 1.
    """
    text = re.sub(
        r"\\Ind\[([^\]]+)\]",
        lambda match: rf"\mathbb{{1}}\left({match.group(1)}\right)",
        text,
    )
    return re.sub(r"\\Ind\b", r"\\mathbb{1}", text)


def normalize_expectation(text: str) -> str:
    r"""Expand the book's ``\E`` macro, including its optional arguments."""
    text = re.sub(
        r"\\E\[([^\]]+)\]\[([^\]]+)\]",
        lambda match: (
            rf"\mathbb{{E}}_{{{match.group(1)}}}\left[{match.group(2)}\right]"
        ),
        text,
    )
    text = re.sub(
        r"\\E\[([^\]]+)\]",
        lambda match: rf"\mathbb{{E}}\left[{match.group(1)}\right]",
        text,
    )
    return re.sub(r"\\E\b", r"\\mathbb{E}", text)


def normalize_kl(text: str) -> str:
    r"""Expand the book's ``\KL{p}{q}`` macro to standard KL-divergence LaTeX."""
    return _replace_braced_command(
        text,
        r"\KL",
        2,
        lambda args: rf"D_{{KL}}\left({args[0]}\middle\|{args[1]}\right)",
    )


def normalize_capital_tilde(text: str) -> str:
    r"""Normalize non-standard ``\Tilde{...}`` to standard ``\widetilde{...}``."""
    return _replace_braced_command(
        text,
        r"\Tilde",
        1,
        lambda args: rf"\widetilde{{{args[0]}}}",
    )


def _roman_numeral(number: int) -> str:
    """Return a lowercase Roman numeral for a positive integer."""
    if number <= 0:
        return str(number)
    values = (
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    )
    pieces: list[str] = []
    remainder = number
    for value, numeral in values:
        count, remainder = divmod(remainder, value)
        pieces.append(numeral * count)
    return "".join(pieces)


def normalize_roman_numerals(text: str) -> str:
    r"""Replace TeX ``\romannumeral<number>`` with literal lowercase numerals."""
    return re.sub(
        r"\\romannumeral\s*(\d+)",
        lambda match: _roman_numeral(int(match.group(1))),
        text,
    )


def normalize_notation(text: str) -> str:
    """Expand the conservative subset of global book notation macros."""
    text = normalize_indicator(text)
    text = normalize_expectation(text)
    text = normalize_kl(text)
    text = normalize_capital_tilde(text)
    text = normalize_roman_numerals(text)
    for source, replacement in COMPOUND_REPLACEMENTS:
        text = text.replace(source, replacement)
    for source, replacement in SIMPLE_MATH_REPLACEMENTS:
        text = text.replace(source, replacement)
    return text


def strip_tex_comments(text: str) -> str:
    """Remove TeX comments without introducing artificial paragraph breaks.

    A TeX percent comment consumes the remainder of its physical line, including
    the end-of-line token. Removing only the comment text leaves runs of blank
    lines behind, which a Markdown converter can incorrectly interpret as new
    paragraphs. Consume the commented newline as TeX does; any genuine blank
    lines already present in the source remain untouched.
    """
    return re.sub(r"(?<!\\)%[^\n]*(?:\n|$)", "", text)


def normalize_typography(text: str) -> str:
    """Drop non-semantic TeX presentation commands while retaining their text."""
    text = re.sub(
        r"\\chapterinitial\{([^{}]+)\}\{([^{}]+)\}",
        lambda match: match.group(1) + match.group(2),
        text,
    )
    for command in TYPOGRAPHIC_COMMANDS:
        text = text.replace(command, "")
    return text


def normalize_latex(text: str) -> str:
    """Normalize comments, notation, and non-semantic typography before MyST conversion."""
    text = strip_tex_comments(text)
    text = normalize_notation(text)
    text = normalize_typography(text)
    return text
