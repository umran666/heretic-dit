"""Aggregate sweep results into paper tables and figures."""

from heretic_dit.reporting.aggregate import (
    RunRecord,
    group_mean,
    load_results,
    main_table_rows,
    pareto_front,
    per_erasure_breakdown,
)
from heretic_dit.reporting.figures import (
    plot_per_erasure_breakdown,
    plot_pareto,
    plot_proxy_correlation,
)
from heretic_dit.reporting.tables import (
    breakdown_table_latex,
    main_table_latex,
    main_table_markdown,
)

__all__ = [
    "RunRecord",
    "group_mean",
    "load_results",
    "main_table_rows",
    "pareto_front",
    "per_erasure_breakdown",
    "plot_per_erasure_breakdown",
    "plot_pareto",
    "plot_proxy_correlation",
    "breakdown_table_latex",
    "main_table_latex",
    "main_table_markdown",
]
