"""Convert paired-metric experiment tables to rectangular MyST tables."""

from __future__ import annotations

import re

from latex_normalize import normalize_latex
from myst_structures import ExtractedStructure, _clean_inline_tex, parse_braced


def extract_metric_tables(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Flatten two grouped metric headers and pad format separator rows.

    Other table layouts remain in the normal MyST conversion pipeline.
    """
    structures: list[ExtractedStructure] = []

    def replace(match: re.Match[str]) -> str:
        body = match.group(0)
        tabular = re.search(
            r"\\begin\{tabular\}\{l\|cc\|cc\}(.*?)\\end\{tabular\}",
            body,
            re.DOTALL,
        )
        if tabular is None:
            return body
        rows = [
            row.strip()
            for row in re.sub(r"\\hline\b", "", tabular.group(1)).split(r"\\")
            if row.strip()
        ]
        if len(rows) < 2:
            return body
        metrics = re.findall(r"\\multicolumn\{2\}\{[^{}]*\}\{", rows[0])
        if len(metrics) != 2:
            return body
        metric_titles = [
            parse_braced(rows[0], m.end() - 1)[0]
            for m in re.finditer(r"\\multicolumn\{2\}\{[^{}]*\}\{", rows[0])
        ]

        def clean(cell: str) -> str:
            cell = normalize_latex(cell)
            cell = re.sub(r"\{(\s*\$\\pm\$\s*[\d.]+)\}", r"\1", cell)
            cell = _clean_inline_tex(cell).replace(r"\%", "%")
            # Keep value and standard error together in one table cell.
            cell = cell.replace(r"$\pm$", " ± ").replace("|", r"\|")
            return re.sub(r"\s+", " ", cell).strip()

        datasets = rows[1].split("&")[1:]
        if len(datasets) != 4:
            return body
        headers = ["Method"] + [
            clean(f"{metric_titles[i // 2]} — {dataset}")
            for i, dataset in enumerate(datasets)
        ]
        caption_pos = body.find(r"\caption")
        caption = parse_braced(body, body.index("{", caption_pos))[0]
        label = re.search(r"\\label\{([^{}]+)\}", body)
        if label is None:
            return body
        lines = [f":::{{table}} {clean(caption)}", f":label: {label.group(1)}", ""]
        lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * 5) + " |"]
        for row in rows[2:]:
            cells = [clean(cell) for cell in row.split("&")]
            if len(cells) > 5:
                raise ValueError("Unexpected number of cells in paired-metric table")
            cells += [""] * (5 - len(cells))
            if not any(cells[1:]):
                cells[0] = f"**{cells[0]}**"
            lines.append("| " + " | ".join(cells) + " |")
        lines.append(":::")
        token = f"BDLTABLEPLACEHOLDER{len(structures):04d}"
        structures.append(ExtractedStructure(token, "\n".join(lines)))
        return token

    return re.sub(r"\\begin\{table\}.*?\\end\{table\}", replace, text, flags=re.DOTALL), structures


def extract_booktabs_tables(text: str) -> tuple[str, list[ExtractedStructure]]:
    """Convert simple rectangular tabularx/booktabs tables with one header row.

    Leave other layouts to the existing converters. Reject ambiguous cell
    splitting rather than silently changing table contents.
    """
    structures: list[ExtractedStructure] = []

    def replace(match: re.Match[str]) -> str:
        body = match.group(0)
        tabular = re.search(r"\\begin\{tabularx\}", body)
        if tabular is None or r"\midrule" not in body:
            return body
        if re.search(r"\\(?:multicolumn|multirow|begin\{(?:aligned|matrix))", body):
            return body
        _, pos = parse_braced(body, tabular.end())
        _, pos = parse_braced(body, pos)
        end = body.index(r"\end{tabularx}", pos)
        header, data = body[pos:end].split(r"\midrule", 1)

        def rows(value: str) -> list[list[str]]:
            value = re.sub(r"\\(?:toprule|bottomrule)\b", "", value)
            return [
                [_clean_inline_tex(normalize_latex(cell)).replace("|", r"\|")
                 for cell in re.split(r"(?<!\\)&", row)]
                for row in value.split(r"\\") if row.strip()
            ]

        headers = rows(header)
        values = rows(data)
        if len(headers) != 1 or not values:
            raise ValueError("Expected one booktabs header row and nonempty data")
        width = len(headers[0])
        if any(len(row) != width for row in values):
            raise ValueError("Nonrectangular booktabs table")
        caption = re.search(r"\\caption\s*\{", body)
        label = re.search(r"\\label\{([^{}]+)\}", body)
        title = _clean_inline_tex(normalize_latex(
            parse_braced(body, caption.end() - 1)[0]
        )) if caption else ""
        lines = [f":::{{table}} {title}".rstrip()]
        if label:
            lines.append(f":label: {label.group(1)}")
        lines.extend(["", "| " + " | ".join(headers[0]) + " |",
                      "| " + " | ".join(["---"] * width) + " |"])
        lines.extend("| " + " | ".join(row) + " |" for row in values)
        lines.append(":::")
        token = f"BDLBOOKTABSTABLEPLACEHOLDER{len(structures):04d}"
        structures.append(ExtractedStructure(token, "\n".join(lines)))
        return token

    return re.sub(r"\\begin\{table\}.*?\\end\{table\}", replace, text, flags=re.DOTALL), structures
