"""Shared budget handling and cost accounting for all baselines."""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping

import torch

from heretic_dit.interfaces import RecoveryResult

__all__ = ["CostTracker", "require_budget_key", "merge_cost", "BUDGET_DOC"]

#: Documentation of the budget contract shared by every baseline. Kept as a
#: string so it can be surfaced in docs and error messages.
BUDGET_DOC = """
Common budget keys understood by every baseline runner:
    validator:      DeterministicGenerativeValidator (required) -- scores recovery.
    adapter_factory: Callable[[model], GenerationAdapter] (required) -- wraps a
                     (possibly recovered) model for generation.
    seed:            int run seed (default 0).
    num_samples:     int validation images per concept (default 16).
    steps:           int training step budget (sweep parameter for TI/LoRA).
    lr:              float learning rate (default 5e-4).
    backend:         pluggable RecoveryBackend implementing the training loop.
"""

# Keys every baseline needs to run at all.
COMMON_REQUIRED_KEYS = ("validator", "adapter_factory")


def require_budget_key(budget: Mapping[str, Any], key: str, runner: str) -> Any:
    """Fetch a required budget entry or raise an actionable error."""
    if key not in budget or budget[key] is None:
        raise KeyError(
            f"{runner} requires budget[{key!r}]. {BUDGET_DOC}"
        )
    return budget[key]


class CostTracker:
    """Wall-clock + peak-VRAM tracker used around each baseline's work."""

    def __init__(self) -> None:
        self._start: float = 0.0

    def __enter__(self) -> "CostTracker":
        self._start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._wall_clock = time.perf_counter() - self._start

    @property
    def wall_clock_sec(self) -> float:
        return getattr(self, "_wall_clock", 0.0)

    @property
    def peak_vram_mb(self) -> float:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return float(torch.cuda.max_memory_allocated()) / (1024.0**2)
        return 0.0

    def snapshot(self, trainable_params: int = 0, sample_count: int = 0) -> Dict[str, Any]:
        """Standardized cost block shared by all methods and baselines."""
        return {
            "wall_clock_sec": float(self.wall_clock_sec),
            "peak_vram_mb": float(self.peak_vram_mb),
            "trainable_params": int(trainable_params),
            "sample_count": int(sample_count),
        }


def merge_cost(result: RecoveryResult, cost: Mapping[str, Any]) -> RecoveryResult:
    """Return ``result`` with ``cost`` merged into its cost block."""
    merged = dict(result.cost)
    merged.update(dict(cost))
    result.cost = merged
    return result
