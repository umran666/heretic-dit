"""Paper figures: Pareto fronts, proxy-vs-real correlation, per-erasure bars.

Matplotlib is used with the non-interactive Agg backend; every figure is
written twice (PNG for review, PDF for the paper).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from heretic_dit.reporting.aggregate import pareto_front, per_erasure_breakdown  # noqa: E402

__all__ = ["plot_pareto", "plot_proxy_correlation", "plot_per_erasure_breakdown"]

_BASELINE_METHODS = {
    "no-edit",
    "random-projection",
    "unerased-reference",
    "textual-inversion",
    "lora-finetune-recovery",
    "full-finetune-recovery",
}


def _save(fig: plt.Figure, path_base: str | Path) -> None:
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    records: Sequence[Any],
    path_base: str | Path,
    title: str = "Recovery vs. quality loss (lower-right is better)",
) -> List[Tuple[str, str]]:
    """Pareto-front scatter of (quality loss, recovery), baselines overlaid.

    X axis is quality loss (``metrics['fid']`` when present, else drift),
    Y axis is recovery. The main method's frontier is drawn as a step line;
    baseline points are overlaid for the same family. Returns the points
    that made the frontier of the main method.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    frontier_methods: List[Tuple[str, str]] = []
    families: Dict[Tuple[str, str], List[Tuple[float, float, str]]] = {}
    for record in records:
        loss = record.metrics.get("fid")
        if loss is None:
            loss = record.drift_score
        families.setdefault((record.erasure_method, record.base_model), []).append(
            (float(loss), record.recovery_score, record.method)
        )
    for (erasure_method, base_model), points in sorted(families.items()):
        main_points = [p for p in points if p[2] not in _BASELINE_METHODS]
        for loss, recovery, method in points:
            is_baseline = method in _BASELINE_METHODS
            ax.scatter(
                loss,
                recovery,
                marker="s" if is_baseline else "o",
                alpha=0.7,
                label=method if method not in ax.get_legend_handles_labels()[1] else None,
            )
        if main_points:
            indices = pareto_front([(r, l) for l, r, _ in main_points])
            front = [main_points[i] for i in indices]
            front.sort(key=lambda p: p[0])
            if len(front) > 1:
                ax.step(
                    [p[0] for p in front],
                    [p[1] for p in front],
                    where="post",
                    color="crimson",
                    linewidth=2,
                )
            for loss, _recovery, method in front:
                frontier_methods.append((erasure_method, method))
    ax.set_xlabel("Quality loss (FID on neutral prompts)")
    ax.set_ylabel("Concept recovery rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    _save(fig, path_base)
    return frontier_methods


def plot_proxy_correlation(
    report: Any,
    trials: Optional[Sequence[Any]] = None,
    path_base: str | Path = "figures/proxy_validity",
    title: str = "Proxy vs. real metrics",
) -> Dict[str, float]:
    """Scatter of proxy vs. real recovery (and utility), annotated with Spearman rho.

    ``report`` is a ``ProxyValidityReport``; ``trials`` (optional) supplies
    the scatter points as ``ProxyTrial`` objects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    if trials:
        weight_r = report.utility_recovery_weight
        weight_d = report.utility_drift_weight
        axes[0].scatter(
            [t.proxy_recovery for t in trials],
            [t.real_recovery for t in trials],
            alpha=0.7,
            s=18,
        )
        axes[1].scatter(
            [weight_r * t.proxy_recovery - weight_d * t.proxy_drift for t in trials],
            [t.real_recovery - t.real_quality_loss for t in trials],
            alpha=0.7,
            s=18,
        )
        axes[1].set_xlabel("Proxy utility")
        axes[1].set_ylabel("Real utility")
    axes[0].set_xlabel("Proxy recovery")
    axes[0].set_ylabel("Real recovery")
    axes[0].set_title(f"Recovery: $\\rho$ = {report.recovery_rho:.3f}")
    axes[1].set_title(f"Utility: $\\rho$ = {report.utility_rho:.3f}")
    fig.suptitle(title)
    _save(fig, path_base)
    return {
        "recovery_rho": report.recovery_rho,
        "quality_rho": report.quality_rho,
        "utility_rho": report.utility_rho,
    }


def plot_per_erasure_breakdown(
    records: Sequence[Any],
    path_base: str | Path = "figures/erasure_breakdown",
    title: str = "Recovery by method, per erasure method",
) -> Dict[str, Dict[str, float]]:
    """Grouped bars: recovery per recovery method, grouped by erasure method.

    Erasure methods fail differently; this figure keeps them apart instead of
    averaging over them.
    """
    breakdown = per_erasure_breakdown(records)
    erasure_methods = list(breakdown)
    methods = sorted({m for by_method in breakdown.values() for m in by_method})
    fig, ax = plt.subplots(figsize=(1.1 * len(methods) + 1.5, 4.0))
    group_width = 0.8 / max(1, len(erasure_methods))
    for index, erasure_method in enumerate(erasure_methods):
        offsets = [i + index * group_width for i in range(len(methods))]
        values = [
            breakdown[erasure_method].get(method, {}).get("recovery", 0.0)
            for method in methods
        ]
        ax.bar(offsets, values, width=group_width, label=erasure_method)
    ax.set_xticks([i + 0.8 * group_width * (len(erasure_methods) - 1) / 2 for i in range(len(methods))])
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean concept recovery")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    _save(fig, path_base)
    return {
        erasure: {method: stats_block["recovery"] for method, stats_block in by_method.items()}
        for erasure, by_method in breakdown.items()
    }
