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
