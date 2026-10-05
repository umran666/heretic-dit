"""Result aggregation: loading, grouping, main-table rows, Pareto fronts.

Input is the results directory written by ``scripts/run_experiment.py``: one
JSON per run containing ``run_spec`` and ``result``. Erasure methods are never
averaged away: every aggregation groups by erasure method first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RunRecord",
    "load_results",
    "group_mean",
    "main_table_rows",
    "pareto_front",
    "per_erasure_breakdown",
    "records_to_dicts",
]


@dataclass
class RunRecord:
    """One finished run, flattened for aggregation."""

    run_id: str
    base_model: str
    erasure_method: str
    concept: str
    method: str
    seed: int
    budget: Dict[str, Any] = field(default_factory=dict)
    recovery_score: float = 0.0
    drift_score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)

    def quality_loss(self) -> Optional[float]:
        """Composite real quality loss from the metrics block, if present.

        Convention: quality_loss = normalized FID + LPIPS + CLIP-score drop,
        each min-max normalized within the run's (base_model, erasure_method,
        concept) family by the caller when a composite is needed. Here we
        expose the raw components; tables pick one primary (``fid``) and list
        the rest.
        """
        metrics = {k: v for k, v in self.metrics.items() if isinstance(v, (int, float))}
        return metrics.get("fid")


def load_results(results_dir: str | Path) -> List[RunRecord]:
    """Load every ``result.json`` under ``results_dir`` (recursive)."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    records: List[RunRecord] = []
    for result_path in sorted(results_dir.rglob("result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        spec = payload.get("run_spec", {})
        result = payload.get("result", {})
        records.append(
            RunRecord(
                run_id=spec.get("run_id", result_path.parent.name),
                base_model=spec.get("base_model", ""),
                erasure_method=spec.get("erasure_method", ""),
                concept=spec.get("concept", ""),
                method=spec.get("method", result.get("method", "")),
                seed=int(spec.get("seed", 0)),
                budget=dict(spec.get("budget", {})),
                recovery_score=float(result.get("recovery_score", 0.0)),
                drift_score=float(result.get("drift_score", 0.0)),
                metrics=dict(result.get("metrics", {})),
                cost=dict(result.get("cost", {})),
            )
        )
    return records


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def group_mean(
    records: Sequence[RunRecord],
    key: str,
    value: str = "recovery_score",
) -> Dict[str, float]:
    """Mean of ``value`` grouped by record field ``key``."""
    groups: Dict[str, List[float]] = {}
    for record in records:
        groups.setdefault(getattr(record, key), []).append(_value_of(record, value))
    return {group: _mean(values) for group, values in sorted(groups.items())}


def _value_of(record: RunRecord, value: str) -> float:
    if value == "recovery_score":
        return record.recovery_score
    if value == "drift_score":
        return record.drift_score
    if value.startswith("cost."):
        return float(record.cost.get(value[5:], 0.0))
    return float(record.metrics.get(value, float("nan")))


COST_KEYS = ("wall_clock_sec", "peak_vram_mb", "trainable_params", "sample_count")


def main_table_rows(records: Sequence[RunRecord]) -> List[Dict[str, Any]]:
    """One row per (base_model, erasure_method, method): recovery vs quality vs cost.

    Quality columns (``fid``, ``lpips``, ``clip_delta``) are only filled when
    the corresponding paired quality metrics were attached to the runs.
    Cost columns are means of the standardized cost block; ``steps`` is
    included when the method swept over training budgets.
    """
    families: Dict[Tuple[str, str, str], List[RunRecord]] = {}
    for record in records:
        families.setdefault((record.base_model, record.erasure_method, record.method), []).append(record)
    rows: List[Dict[str, Any]] = []
    for (base_model, erasure_method, method), group in sorted(families.items()):
        row: Dict[str, Any] = {
            "base_model": base_model,
            "erasure_method": erasure_method,
            "method": method,
            "n_runs": len(group),
            "recovery": _mean([r.recovery_score for r in group]),
            "recovery_std": (
                _std([r.recovery_score for r in group]) if len(group) > 1 else 0.0
            ),
            "drift": _mean([r.drift_score for r in group]),
        }
        for metric_key, column in (
            ("fid", "fid"),
            ("lpips", "lpips"),
            ("clip_score_delta", "clip_delta"),
        ):
            values = [r.metrics[metric_key] for r in group if metric_key in r.metrics]
            row[column] = _mean(values) if values else None
        for cost_key in COST_KEYS:
            row[cost_key] = _mean([float(r.cost.get(cost_key, 0.0)) for r in group])
        steps = [float(r.cost["steps"]) for r in group if "steps" in r.cost]
        row["steps"] = _mean(steps) if steps else None
        rows.append(row)
    return rows


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def pareto_front(
    points: Sequence[Tuple[float, float]],
    maximize_recovery: bool = True,
) -> List[int]:
    """Indices of the Pareto frontier over (recovery, quality_loss) points.

    A point is on the frontier if no other point dominates it (recovery
    greater-or-equal AND quality loss less-or-equal, with at least one strict).
    """
    if not points:
        return []
    for recovery, loss in points:
        if recovery < 0 or loss < 0:
            raise ValueError("Pareto points must be non-negative.")
    front: List[int] = []
    for i, (recovery_i, loss_i) in enumerate(points):
        dominated = False
        for j, (recovery_j, loss_j) in enumerate(points):
            if i == j:
                continue
            better_recovery = recovery_j >= recovery_i
            better_loss = loss_j <= loss_i
            strictly = (recovery_j > recovery_i) or (loss_j < loss_i)
            if better_recovery and better_loss and strictly:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return sorted(front)


def per_erasure_breakdown(records: Sequence[RunRecord]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Nested breakdown: erasure_method -> method -> mean metrics.

    Different erasure methods fail differently, so every table and figure that
    aggregates across erasure methods must expose this breakdown rather than
    a single averaged number.
    """
    families: Dict[str, Dict[str, List[RunRecord]]] = {}
    for record in records:
        families.setdefault(record.erasure_method, {}).setdefault(record.method, []).append(record)
    breakdown: Dict[str, Dict[str, Dict[str, float]]] = {}
    for erasure_method, by_method in sorted(families.items()):
        breakdown[erasure_method] = {}
        for method, group in sorted(by_method.items()):
            breakdown[erasure_method][method] = {
                "recovery": _mean([r.recovery_score for r in group]),
                "drift": _mean([r.drift_score for r in group]),
                "wall_clock_sec": _mean([float(r.cost.get("wall_clock_sec", 0.0)) for r in group]),
                "trainable_params": _mean([float(r.cost.get("trainable_params", 0.0)) for r in group]),
                "sample_count": _mean([float(r.cost.get("sample_count", 0.0)) for r in group]),
                "n_runs": float(len(group)),
            }
    return breakdown


def records_to_dicts(records: Sequence[RunRecord]) -> List[Dict[str, Any]]:
    return [
        {
            "run_id": r.run_id,
            "base_model": r.base_model,
            "erasure_method": r.erasure_method,
            "concept": r.concept,
            "method": r.method,
            "seed": r.seed,
            "recovery_score": r.recovery_score,
            "drift_score": r.drift_score,
            "metrics": dict(r.metrics),
            "cost": dict(r.cost),
        }
        for r in records
    ]
