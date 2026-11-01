"""PDF-only compatibility fixes for the installed MyST 1.10 LaTeX exporter.

Published Markdown is never rewritten by these adaptations. Algorithms use a
supported theorem container in the export copy; the PDF template names that
container Algorithm. Nested figure floats are emitted as standard subfigures,
retaining the source parent label and the native export's panel labels.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def site_references(site: Path) -> dict[str, dict]:
    """Read MyST's own resolved targets and numbers for links outside a PDF."""
    references = json.loads((site / "myst.xref.json").read_text())["references"]
    numbers: dict[str, str] = {}

    def visit(value):
        if isinstance(value, dict):
            if value.get("identifier") and value.get("enumerator") is not None:
                if value.get("type") != "crossReference":
                    numbers[value["identifier"].casefold()] = str(value["enumerator"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for page in (site / "content").glob("*.json"):
        visit(json.loads(page.read_text()))
    return {
        ref["identifier"].casefold(): {**ref, "number": numbers.get(ref["identifier"].casefold())}
        for ref in references if ref.get("identifier")
    }


def repair_references(tex: str, references: dict[str, dict]) -> str:
    local = {label.casefold(): label for label in re.findall(r"\\label\{([^{}]+)\}", tex)}

    def replace(match: re.Match[str]) -> str:
        label = match.group(2)
        if label.casefold() in local:
            return "\\" + match.group(1) + "{" + local[label.casefold()] + "}"
        target = references.get(label.casefold())
        if target is None or target.get("number") is None:
            raise ValueError(f"Unresolved reference in standalone PDF: {label}")
        number = str(target["number"])
        if match.group(1) == "eqref":
            number = "(" + number + ")"
        url = "https://bdlbook.org" + target["url"] + "#" + target["html_id"]
        return rf"\href{{{url}}}{{{number}}}"

    tex = re.sub(r"\\(eqref|ref)\{([^{}]+)\}", replace, tex)
    def href(match: re.Match[str]) -> str:
        url = match.group(1)
        if url.startswith("/"):
            url = "https://bdlbook.org" + url
        # A URL inside a footnote/caption is read as a macro argument first.
        url = re.sub(r"(?<!\\)#", lambda _: r"\#", url)
        return r"\href{" + url + "}"

    tex = re.sub(r"\\href\{([^{}]*)\}", href, tex)
    return re.sub(r"Axiom(?=~\\(?:ref|href)\{)", "Algorithm", tex)


def normalize_math_paragraphs(tex: str) -> str:
    """KaTeX accepts blank math lines that TeX interprets as illegal paragraphs."""
    pattern = (
        r"(\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?)\})"
        r"([\s\S]*?)(\\end\{(?P=env)\})"
    )
    return re.sub(
        pattern,
        lambda m: m.group(1) + re.sub(r"\n\s*\n", "\n", m.group(3)) + m.group(4),
        tex,
    )


def export_markdown(markdown: str) -> str:
    if re.search(r"\{(?:proof|prf):axiom\}", markdown):
        raise ValueError("The PDF algorithm container conflicts with a source axiom")
    markdown = re.sub(r"\{(proof|prf):algorithm\}", r"{\1:axiom}", markdown)
    # MyST 1.10 renders an explicit align environment verbatim, dropping the
    # directive's label. An aligned environment inside its native equation
    # wrapper keeps both the label and the HTML's single block number.
    def labelled_math(match: re.Match[str]) -> str:
        block = match.group()
        if ":label:" not in block:
            return block
        if re.search(r"\\begin\{align\*?\}", block):
            block = re.sub(r"\\(begin|end)\{align\*?\}", r"\\\1{aligned}", block)
            block = re.sub(r"\\(?:notag|nonumber)\b", "", block)
        return block

    markdown = re.sub(
        r"(?m)^(?P<fence>`{3,})\{math\}[^\n]*\n[\s\S]*?^(?P=fence)[ \t]*$",
        labelled_math, markdown,
    )
    return re.sub(r"^\+\+\+\s*$", "", markdown, flags=re.M)


def repair_figures(tex: str, markdown: str) -> str:
    # Options can precede :label:; stop at the first non-option content line.
    labels = []
    for match in re.finditer(r"^:{3,}\{figure\}[^\n]*\n", markdown, re.M):
        options = re.match(r"(?:\s*:[^\n]*\n)*", markdown[match.end():])
        label = re.search(r":label:\s*([^\n]+)", options.group() if options else "")
        if label is None:
            raise ValueError("PDF figures require an explicit parent label")
        labels.append(label.group(1).strip())
    boundary = re.compile(r"\\(begin|end)\{figure\}(?:\[[^\]]*\])?")
    outer: list[tuple[int, int]] = []
    stack: list[int] = []
    for match in boundary.finditer(tex):
        if match.group(1) == "begin":
            stack.append(match.start())
        else:
            start = stack.pop()
            if not stack:
                outer.append((start, match.end()))
    if stack or len(outer) != len(labels):
        raise ValueError("PDF figure structure does not match the source MyST figures")
    for (start, end), label in reversed(list(zip(outer, labels, strict=True))):
        figure = tex[start:end]
        opening = boundary.match(figure)
        assert opening is not None
        inner = figure[opening.end():figure.rfind(r"\end{figure}")]
        children = list(re.finditer(
            r"\\begin\{figure\}(?:\[[^\]]*\])?([\s\S]*?)\\end\{figure\}", inner
        ))
        for child in reversed(children):
            body = child.group(1)
            width = re.search(r"\\includegraphics\[width=([^\]]+)\]", body)
            if width is None:
                raise ValueError("Missing width in PDF figure panel")
            body = body.replace(width.group(), r"\includegraphics[width=\linewidth]", 1)
            panel = (
                r"\begin{subfigure}[t]{" + width.group(1) + "}\n"
                + body.strip() + "\n" + r"\end{subfigure}" + "\n"
            )
            inner = inner[:child.start()] + panel + inner[child.end():]
        inner = re.sub(
            r"\\end\{subfigure\}\s*\\begin\{subfigure\}",
            lambda _: r"\end{subfigure}\hfill\begin{subfigure}", inner,
        )
        panels = list(re.finditer(
            r"\\begin\{subfigure\}\[t\]\{([0-9.]+)\\(?:line|text)width\}", inner
        ))
        if len(panels) > 2:
            # Large online panels may scroll vertically. On paper keep these
            # groups in the familiar two-column 45% layout so captions and
            # plots remain legible instead of shrinking a six-panel column.
            for panel in reversed(panels):
                if float(panel.group(1)) > 0.45:
                    inner = (
                        inner[:panel.start()] + r"\begin{subfigure}[t]{0.45\linewidth}"
                        + inner[panel.end():]
                    )
        # The 1.10 exporter leaks the last child's label into the parent.
        # The final label belongs to the parent caption; restore it explicitly.
        matches = list(re.finditer(r"\\label\{[^{}]+\}", inner))
        if not matches:
            raise ValueError("Missing label in exported PDF figure")
        parent = matches[-1]
        inner = inner[:parent.start()] + rf"\label{{{label}}}" + inner[parent.end():]
        if r"\begin{adjustbox}" not in inner:
            # Long online figures can exceed a printed page. Include the
            # caption in the measured box so it cannot be clipped off-page.
            inner = (
                "\n\\centering\n"
                r"\begin{adjustbox}{max totalsize={\textwidth}{0.9\textheight},center}"
                "\n" + r"\begin{minipage}{\textwidth}" + "\n" + inner
                + "\n" + r"\end{minipage}" + "\n" + r"\end{adjustbox}" + "\n"
            )
        figure = figure[:opening.end()] + inner + r"\end{figure}"
        tex = tex[:start] + figure + tex[end:]
    return tex
