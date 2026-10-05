"""Proxy-vs-real validity analysis: does the search proxy rank edits correctly?

The Optuna search ranks candidate edits by fast proxy metrics (prediction
drift on cached noisy latents). This module is the paper's key sanity check:
for a sample of searched edits we compute *real* generative recovery and
*real* quality loss, then report Spearman rank correlation between proxy and
real metrics. High Spearman on recovery (and on the combined utility) means
the proxy can be trusted; a low number invalidates the search.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "ProxyTrial",
    "ProxyValidityReport",
    "spearman",
    "evaluate_proxy_validity",
    "save_trials",
    "load_trials",
    "report_to_dict",
]


@dataclass
class ProxyTrial:
    """One evaluated edit: its proxy scores and its ground-truth scores.

    Attributes:
        proxy_recovery: proxy concept-recovery estimate in [0, 1] (higher better).
        proxy_drift: proxy collateral-drift estimate (lower better).
        real_recovery: generative held-out recovery rate in [0, 1].
        real_quality_loss: real paired quality loss (FID/LPIPS/CLIP composite);
            0 means no damage, higher is worse.
        trial_id: optional Optuna trial identifier.
        label: optional human-readable label (e.g. erasure method + concept).
        extra: arbitrary per-trial metadata kept alongside results.
    """

    proxy_recovery: float
    proxy_drift: float
    real_recovery: float
    real_quality_loss: float
    trial_id: Optional[int] = None
    label: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProxyTrial":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(frozen=True)
class ProxyValidityReport:
    """Spearman rank-correlation summary between proxy and real metrics."""

    n_trials: int
    recovery_rho: float
    recovery_pvalue: float
    quality_rho: float
    quality_pvalue: float
    utility_rho: float
    utility_pvalue: float
    utility_recovery_weight: float
    utility_drift_weight: float

    @property
    def trustworthy(self) -> bool:
        """Convention for the paper: recovery rho >= 0.7 and utility rho >= 0.7."""
        return self.recovery_rho >= 0.7 and self.utility_rho >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Spearman rank correlation with input validation.

    Returns:
        ``(rho, pvalue)``.

    Raises:
        ValueError: if inputs are empty, length-mismatched, contain non-finite
            values, or one side is constant (correlation undefined).
    """
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: {len(x)} vs {len(y)}.")
    if len(x) < 3:
        raise ValueError("Spearman correlation needs at least 3 paired points.")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x_arr).all() or not np.isfinite(y_arr).all():
        raise ValueError("Spearman inputs must be finite.")
    if np.all(x_arr == x_arr[0]) or np.all(y_arr == y_arr[0]):
        raise ValueError("Spearman correlation is undefined for a constant input.")
    result = stats.spearmanr(x_arr, y_arr)
    rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
    pvalue = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])
    return rho, pvalue


def evaluate_proxy_validity(
    trials: Sequence[ProxyTrial],
    utility_recovery_weight: float = 1.0,
    utility_drift_weight: float = 1.0,
) -> ProxyValidityReport:
    """Correlate proxy metrics with real metrics over a set of evaluated edits.

    Args:
        trials: trials carrying both proxy and real scores. Should span the
            range of proxy values the search actually visits (sample trials
            across the frontier, not only the best ones).
        utility_recovery_weight: weight of recovery in the composite utility.
        utility_drift_weight: weight of drift/quality penalty in the utility.

    Returns:
        A :class:`ProxyValidityReport` with Spearman rho/p for recovery
        (proxy vs real), quality (proxy drift vs real quality loss), and the
        composite utility (weighted proxy utility vs weighted real utility).
    """
    if not trials:
        raise ValueError("evaluate_proxy_validity needs at least one trial.")
    if utility_recovery_weight <= 0.0 or utility_drift_weight <= 0.0:
        raise ValueError("Utility weights must be positive.")

    proxy_recovery = [t.proxy_recovery for t in trials]
    proxy_drift = [t.proxy_drift for t in trials]
    real_recovery = [t.real_recovery for t in trials]
    real_loss = [t.real_quality_loss for t in trials]

    proxy_utility = [
        utility_recovery_weight * r - utility_drift_weight * d
        for r, d in zip(proxy_recovery, proxy_drift)
    ]
    real_utility = [
        utility_recovery_weight * r - utility_drift_weight * l
        for r, l in zip(real_recovery, real_loss)
    ]

    recovery_rho, recovery_p = spearman(proxy_recovery, real_recovery)
    quality_rho, quality_p = spearman(proxy_drift, real_loss)
    utility_rho, utility_p = spearman(proxy_utility, real_utility)

    return ProxyValidityReport(
        n_trials=len(trials),
        recovery_rho=recovery_rho,
        recovery_pvalue=recovery_p,
        quality_rho=quality_rho,
        quality_pvalue=quality_p,
        utility_rho=utility_rho,
        utility_pvalue=utility_p,
        utility_recovery_weight=utility_recovery_weight,
        utility_drift_weight=utility_drift_weight,
    )


def save_trials(trials: Sequence[ProxyTrial], path: str | Path) -> Path:
    """Persist proxy trials to JSON (append-safe single-file format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([t.to_dict() for t in trials], handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_trials(path: str | Path) -> List[ProxyTrial]:
    """Load proxy trials from JSON written by :func:`save_trials`."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payloads = json.load(handle)
    return [ProxyTrial.from_dict(p) for p in payloads]


def report_to_dict(report: ProxyValidityReport) -> Dict[str, Any]:
    return report.to_dict()
