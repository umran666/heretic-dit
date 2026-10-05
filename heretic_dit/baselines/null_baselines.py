"""Null baselines: no edit, random-direction projection, unerased reference.

These anchor the recovery scale:
    * ``no-edit``             -- the erased model as-is (lower bound).
    * ``random-projection``   -- a random orthonormal subspace with *matched*
                                 alpha, to show the found subspace matters.
    * ``unerased-reference``  -- the unerased base model (upper bound).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch

from heretic_dit.baselines.common import CostTracker, merge_cost, require_budget_key
from heretic_dit.eval.validator import resolve_adapter
from heretic_dit.interfaces import EditSpec, RecoveryResult

__all__ = [
    "NoEditBaseline",
    "RandomProjectionBaseline",
    "UnerasedReferenceBaseline",
    "make_null_baseline",
]


class NoEditBaseline:
    """Evaluates the erased model exactly as-is. Lower bound on recovery."""

    method_name = "no-edit"

    def run(self, erased_model: Any, concept: str, budget: Dict[str, Any]) -> RecoveryResult:
        validator = require_budget_key(budget, "validator", self.method_name)
        adapter_factory = require_budget_key(budget, "adapter_factory", self.method_name)
        adapter = resolve_adapter(adapter_factory(erased_model))
        with CostTracker() as tracker:
            result = validator.validate(
                adapter,
                concept,
                int(budget.get("num_samples", 16)),
                int(budget.get("seed", 0)),
            )
        result.method = self.method_name
        return merge_cost(result, tracker.snapshot(trainable_params=0))


class RandomProjectionBaseline:
    """Projects cross-attention weights onto a *random* orthonormal subspace.

    Uses the same ``alpha``, mode, and side as the searched edit so the
    comparison isolates the choice of subspace direction. The actual weight
    surgery is delegated to ``budget["apply_edit"]`` (a callable owning the
    core projector); this baseline only constructs the random ``EditSpec``.
    """

    method_name = "random-projection"

    def __init__(self, dim: int = 8, mode: str = "orthogonal", side: str = "input") -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}.")
        self._dim = dim
        self._mode = mode
        self._side = side

    def run(self, erased_model: Any, concept: str, budget: Dict[str, Any]) -> RecoveryResult:
        apply_edit: Callable[[Any, EditSpec], Any] = require_budget_key(
            budget, "apply_edit", self.method_name
        )
        validator = require_budget_key(budget, "validator", self.method_name)
        adapter_factory = require_budget_key(budget, "adapter_factory", self.method_name)
        seed = int(budget.get("seed", 0))
        alpha = float(budget.get("alpha", 1.0))

        generator = torch.Generator().manual_seed(seed)
        model_dim = int(budget.get("model_dim", 0))
        if model_dim <= 0:
            raise KeyError(
                f"{self.method_name} requires budget['model_dim'] (the cross-attention "
                "input dimension) to draw a matched random subspace."
            )
        raw = torch.randn(model_dim, self._dim, generator=generator)
        basis, _ = torch.linalg.qr(raw)
        edit_spec = EditSpec(
            layers=tuple(budget.get("layers", ())),
            alpha=alpha,
            mode=self._mode,  # type: ignore[arg-type]
            side=self._side,  # type: ignore[arg-type]
            subspace=basis,
            metadata={"source": "random", "concept": concept, "seed": seed},
        )
        with CostTracker() as tracker:
            projected_model = apply_edit(erased_model, edit_spec)
            adapter = resolve_adapter(adapter_factory(projected_model))
            result = validator.validate(
                adapter,
                concept,
                int(budget.get("num_samples", 16)),
                seed,
            )
        result.method = self.method_name
        cost = tracker.snapshot(trainable_params=0)
        cost["alpha"] = alpha
        cost["subspace_dim"] = int(self._dim)
        return merge_cost(result, cost)


class UnerasedReferenceBaseline:
    """Evaluates the unerased base model. Upper bound on recovery."""

    method_name = "unerased-reference"

    def run(self, erased_model: Any, concept: str, budget: Dict[str, Any]) -> RecoveryResult:
        validator = require_budget_key(budget, "validator", self.method_name)
        adapter_factory = require_budget_key(budget, "adapter_factory", self.method_name)
        reference_model = budget.get("reference_model", erased_model)
        adapter = resolve_adapter(adapter_factory(reference_model))
        with CostTracker() as tracker:
            result = validator.validate(
                adapter,
                concept,
                int(budget.get("num_samples", 16)),
                int(budget.get("seed", 0)),
            )
        result.method = self.method_name
        return merge_cost(result, tracker.snapshot(trainable_params=0))


def make_null_baseline(kind: str) -> Any:
    """Factory: ``kind`` in {``no-edit``, ``random-projection``, ``unerased-reference``}."""
    kinds = {
        "no-edit": NoEditBaseline,
        "random-projection": RandomProjectionBaseline,
        "unerased-reference": UnerasedReferenceBaseline,
    }
    if kind not in kinds:
        raise KeyError(f"Unknown null baseline {kind!r}; choose from {sorted(kinds)}.")
    return kinds[kind]()
