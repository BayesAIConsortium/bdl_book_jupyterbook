#!/usr/bin/env python3
"""Shared normalization for book-specific LaTeX before MyST conversion.

This module handles reusable notation expansion and non-semantic typographic
cleanup. Semantic structures such as algorithms, theorems, proofs, and chapter
headings belong in myst_structures.py or other shared structural modules.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from text_normalize import normalize_tex_text_accents


COMPOUND_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # The dataset macro is mathematical even inside a TeX text subscript.
    (r"\text{\traindata}", r"{\mathcal{D}}"),
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
    (r"\numPar", "R"),
    (r"\idxPar", "r"),
    (r"\sX", r"\mathsf{X}"),
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
    r"\noindent",
    r"\scriptsize",
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


def _replace_group_declaration(text: str, declaration: str, render: Callable[[str], str]) -> str:
    r"""Replace ``{\declaration ...}`` groups while preserving TeX argument braces."""
    pieces: list[str] = []
    pos = 0
    needle = "{" + declaration
    while True:
        group_pos = text.find(needle, pos)
        if group_pos < 0:
            pieces.append(text[pos:])
            break

        after_declaration = group_pos + len(needle)
        if after_declaration < len(text) and text[after_declaration].isalpha():
            pieces.append(text[pos:after_declaration])
            pos = after_declaration
            continue

        pieces.append(text[pos:group_pos])
        try:
            group, end = _parse_braced(text, group_pos)
        except ValueError:
            pieces.append(text[group_pos : group_pos + 1])
            pos = group_pos + 1
            continue

        remainder = group[len(declaration) :].lstrip()
        rendered = render(remainder)
        # In ``\paragraph{\it Title}``, the declaration group is also the
        # command's required argument. Preserve those outer braces; otherwise
        # the normalization would incorrectly produce ``\paragraph\textit{...}``.
        is_command_argument = re.search(
            r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?\s*$", text[:group_pos]
        )
        pieces.append("{" + rendered + "}" if is_command_argument else rendered)
        pos = end
    return "".join(pieces)


def _normalize_em_declarations(text: str, *, inside_emphasis: bool = False) -> str:
    r"""Convert legacy ``{\em ...}`` groups to ``\emph{...}``.

    Nested ``\em`` declarations are redundant in TeX. Flatten them while already
    inside emphasis so downstream Markdown conversion cannot create mismatched
    nested emphasis delimiters. Respect TeX control-word boundaries so commands
    such as ``\emph`` are never mistaken for the legacy ``\em`` declaration.
    """
    pieces: list[str] = []
    pos = 0
    needle = r"{\em"
    while True:
        group_pos = text.find(needle, pos)
        if group_pos < 0:
            pieces.append(text[pos:])
            break

        after_em = group_pos + len(needle)
        if after_em < len(text) and text[after_em].isalpha():
            pieces.append(text[pos:after_em])
            pos = after_em
            continue

        pieces.append(text[pos:group_pos])
        try:
            group, end = _parse_braced(text, group_pos)
        except ValueError:
            pieces.append(text[group_pos : group_pos + 1])
            pos = group_pos + 1
            continue

        body = group[len(r"\em") :].lstrip()
        body = _normalize_em_declarations(body, inside_emphasis=True)
        if inside_emphasis:
            pieces.append(body)
        else:
            pieces.append(r"\emph{" + body + "}")
        pos = end
    return "".join(pieces)


def flatten_nested_emphasis(text: str, *, inside_emphasis: bool = False) -> str:
    r"""Flatten nested italic commands before MyST emits Markdown delimiters."""
    pattern = re.compile(r"\\(?:emph|textit)\s*\{")
    pieces: list[str] = []
    pos = 0
    while match := pattern.search(text, pos):
        body, end = _parse_braced(text, match.end() - 1)
        body = flatten_nested_emphasis(body, inside_emphasis=True)
        pieces.append(text[pos:match.start()])
        pieces.append(body if inside_emphasis else match.group(0) + body + "}")
        pos = end
    pieces.append(text[pos:])
    return "".join(pieces)


def _strip_grouped_color(text: str) -> str:
    r"""Remove ``{\color{...} ...}`` styling while preserving the group body.

    Some legacy source spans a color declaration across large structural blocks.
    If that outer group is not balanced in the current conversion fragment, drop
    the purely typographic opening declaration rather than leaking a literal
    opening brace into generated Markdown.
    """
    pieces: list[str] = []
    pos = 0
    needle = r"{\color"
    while True:
        group_pos = text.find(needle, pos)
        if group_pos < 0:
            pieces.append(text[pos:])
            break
        pieces.append(text[pos:group_pos])
        try:
            group, end = _parse_braced(text, group_pos)
            _, cursor = _parse_braced(group, len(r"\color"))
        except ValueError:
            try:
                _, cursor = _parse_braced(text, group_pos + 1 + len(r"\color"))
            except ValueError:
                pieces.append(text[group_pos : group_pos + 1])
                pos = group_pos + 1
                continue
            pos = cursor
            continue
        pieces.append(group[cursor:].lstrip())
        pos = end
    return "".join(pieces)


def normalize_indicator(text: str) -> str:
    r"""Expand the book's indicator-function macro to standard LaTeX."""
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
        lambda match: rf"\mathbb{{E}}_{{{match.group(1)}}}\left[{match.group(2)}\right]",
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


def normalize_norm(text: str) -> str:
    r"""Expand the book's paired-delimiter ``\norm{...}`` macro."""
    return _replace_braced_command(
        text,
        r"\norm",
        1,
        lambda args: rf"\left\lVert {args[0]} \right\rVert",
    )


def normalize_differential(text: str) -> str:
    r"""Expand the book's ``\rd`` infinitesimal macro."""
    return re.sub(r"\\rd(?![A-Za-z])", r"\\,\\mathrm{d}", text)


def normalize_variance_macro(text: str) -> str:
    r"""Expand the book's ``\Var`` symbol while preserving following scripts."""
    return re.sub(r"\\Var(?![A-Za-z])", r"\\mathbb{V}", text)


def normalize_variance(text: str) -> str:
    r"""Normalize legacy variance notation to standard LaTeX."""
    text = text.replace(
        r"\mathbb{V}\textrm{\emph{ar}}",
        r"\operatorname{Var}",
    )
    return text.replace(
        r"\mathbb{V}\textrm{ar}",
        r"\operatorname{Var}",
    )


def normalize_eqnarray(text: str) -> str:
    r"""Replace legacy ``eqnarray`` environments with ``align`` equivalents.

    This deliberately changes only the environment names. Existing alignment
    markers, labels, ``\nonumber`` commands, and line breaks are preserved.
    """
    text = text.replace(r"\begin{eqnarray*}", r"\begin{align*}")
    text = text.replace(r"\end{eqnarray*}", r"\end{align*}")
    text = text.replace(r"\begin{eqnarray}", r"\begin{align}")
    return text.replace(r"\end{eqnarray}", r"\end{align}")


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


def _replace_simple_math_macros(text: str) -> str:
    """Replace simple book macros without merging adjacent TeX control words."""
    replacements = dict(SIMPLE_MATH_REPLACEMENTS)
    pattern = re.compile(
        "|".join(
            re.escape(source)
            for source in sorted(replacements, key=len, reverse=True)
        )
    )

    def replace(match: re.Match[str]) -> str:
        replacement = replacements[match.group(0)]
        if (
            replacement
            and replacement[0].isalpha()
            and match.start() > 0
            and text[match.start() - 1].isalpha()
        ):
            return " " + replacement
        return replacement

    return pattern.sub(replace, text)


def normalize_notation(text: str) -> str:
    """Expand the conservative subset of global book notation macros."""
    text = normalize_indicator(text)
    text = normalize_expectation(text)
    text = normalize_kl(text)
    text = normalize_capital_tilde(text)
    text = normalize_norm(text)
    text = normalize_differential(text)
    text = normalize_variance_macro(text)
    text = normalize_variance(text)
    text = normalize_roman_numerals(text)
    for source, replacement in COMPOUND_REPLACEMENTS:
        text = text.replace(source, replacement)
    return _replace_simple_math_macros(text)


def normalize_pre_extraction(text: str) -> str:
    r"""Normalize source syntax needed before structural extraction.

    In the web book, a figure's enclosing content width plays the same role as
    TeX ``\columnwidth``. Mapping it to ``\textwidth`` lets the shared figure
    extractor preserve author-specified fractional widths consistently.
    """
    return normalize_eqnarray(text.replace(r"\columnwidth", r"\textwidth"))


def strip_tex_comments(text: str) -> str:
    """Remove TeX comments without introducing artificial paragraph breaks."""
    return re.sub(r"(?<!\\)%[^\n]*(?:\n|$)", "", text)


def normalize_typography(text: str) -> str:
    """Drop non-semantic TeX presentation commands while retaining their text."""
    text = re.sub(
        r"\\chapterinitial\{([^{}]+)\}\{([^{}]+)\}",
        lambda match: match.group(1) + match.group(2),
        text,
    )

    # Remove no-argument presentation commands before interpreting grouped
    # declarations. This prevents ``\noindent{\bf ...}`` from being mistaken
    # for a command whose required argument is the following brace group.
    for command in TYPOGRAPHIC_COMMANDS:
        text = text.replace(command, "")

    # Color is purely presentational in the source; preserve its contents only.
    text = _replace_braced_command(text, r"\textcolor", 2, lambda args: args[1])
    text = _strip_grouped_color(text)

    # Preserve intended emphasis while replacing legacy declaration syntax with
    # standard LaTeX that MyST already understands. Nested ``\em`` declarations
    # are flattened because repeating the same emphasis has no extra semantics.
    text = _normalize_em_declarations(text)
    text = _replace_group_declaration(text, r"\bf", lambda body: rf"\textbf{{{body}}}")
    text = _replace_group_declaration(text, r"\it", lambda body: rf"\textit{{{body}}}")
    return flatten_nested_emphasis(text)


def normalize_latex(text: str) -> str:
    """Normalize comments, notation, and non-semantic typography before MyST conversion."""
    text = strip_tex_comments(text)
    text = normalize_tex_text_accents(text)
    text = normalize_notation(text)
    text = normalize_typography(text)
    return text
