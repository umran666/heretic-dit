"""LaTeX and Markdown table writers for the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from heretic_dit.reporting.aggregate import main_table_rows, per_erasure_breakdown

__all__ = ["main_table_latex", "main_table_markdown", "breakdown_table_latex"]

_METHOD_LABELS = {
    "heretic-dit": "Heretic-DiT (ours)",
    "no-edit": "No edit (lower bound)",
    "random-projection": "Random projection",
    "unerased-reference": "Unerased reference",
    "textual-inversion": "Textual inversion",
    "lora-finetune-recovery": "LoRA fine-tune",
    "full-finetune-recovery": "Full fine-tune",
}


def _escape(text: Any) -> str:
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%").replace("$", r"\$")


def _label(method: str) -> str:
    return _METHOD_LABELS.get(method, method)


def _fmt(value: Any, precision: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def main_table_latex(
    records: Sequence[Any],
    path: Optional[str | Path] = None,
    caption: str = "Concept recovery vs.\\ quality loss vs.\\ cost per method.",
    label: str = "tab:main",
) -> str:
    """Main paper table: booktabs LaTeX, one row per method per erasure method."""
    rows = main_table_rows(records)
    header = (
        "Erasure & Method & Recovery & Drift & FID & LPIPS & "
        r"\cost{wall} (s) & \cost{params} & \cost{samples} \\"
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    current_erasure: Optional[str] = None
    for row in rows:
        if row["erasure_method"] != current_erasure:
            current_erasure = row["erasure_method"]
            lines.append(r"\multicolumn{9}{l}{\textit{" + _escape(current_erasure) + r" erasure}}\\")
            lines.append(r"\midrule")
        lines.append(
            " & ".join(
                [
                    "",
                    _escape(_label(row["method"])),
                    _fmt(row["recovery"]),
                    _fmt(row["drift"]),
                    _fmt(row["fid"]),
                    _fmt(row["lpips"]),
                    _fmt(row["wall_clock_sec"], 1),
                    _fmt(row["trainable_params"], 0),
                    _fmt(row["sample_count"], 0),
                ]
            )
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    return text


def main_table_markdown(records: Sequence[Any], path: Optional[str | Path] = None) -> str:
    """Markdown variant of the main table for READMEs and reports."""
    rows = main_table_rows(records)
    columns = [
        "erasure_method",
        "method",
        "recovery",
        "drift",
        "fid",
        "lpips",
        "wall_clock_sec",
        "trainable_params",
        "sample_count",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = [
            _fmt(row[column], 3) if not isinstance(row[column], str) else row[column]
            for column in columns
        ]
        cells[1] = _label(row["method"])
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    return text


def breakdown_table_latex(
    records: Sequence[Any],
    path: Optional[str | Path] = None,
    caption: str = "Per-erasure-method breakdown of recovery by method.",
    label: str = "tab:breakdown",
) -> str:
    """Per-erasure-method breakdown: rows = methods, column blocks = erasure methods.

    Erasure methods fail differently; this table exists so the paper never
    reports a single number averaged over them.
    """
    breakdown = per_erasure_breakdown(records)
    erasure_methods = list(breakdown)
    methods = sorted({m for by_method in breakdown.values() for m in by_method})
    column_block = "r" * len(erasure_methods)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + column_block + "}",
        r"\toprule",
        " & ".join([""] + [r"\multicolumn{1}{c}{" + _escape(e) + "}" for e in erasure_methods]) + r"\\",
        " & ".join(["Method"] + [r"\multicolumn{1}{c}{Recovery}" for _ in erasure_methods]) + r"\\",
        r"\midrule",
    ]
    for method in methods:
        cells = [_escape(_label(method))]
        for erasure_method in erasure_methods:
            stats_block = breakdown[erasure_method].get(method)
            cells.append(_fmt(stats_block["recovery"]) if stats_block else "--")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    text = "\n".join(lines) + "\n"
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    return text
