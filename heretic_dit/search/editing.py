"""Weight-edit application with snapshot/restore guarantees.

The search loop applies one candidate edit per trial to the *live* model.  Two
properties matter:

* **Speed** - only the touched weight tensors are cloned, never the whole model.
* **Safety** - restoring is unconditional.  :func:`applied_edit` restores from
  snapshots in a ``finally`` block, so a projector that raises mid-way (e.g. a
  singular covariance) cannot leave the model corrupted.

The per-layer scaling uses a Heretic-style kernel rather than one free Optuna
parameter per layer, keeping the search dimension independent of model depth.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from heretic_dit.core.projector import project_weights

__all__ = [
    "ProjectionTarget",
    "InterpolationKernel",
    "LayerEdit",
    "alpha_kernel",
    "resolve_alphas",
    "cross_attention_targets",
    "resolve_targets",
    "apply_alpha",
    "apply_edits",
    "restore_edits",
    "applied_edit",
]

# The search space names the covariance variant "covariance_regularized"; the
# projector calls it "covariance".  Normalize here so the two vocabularies do
# not leak into projector.py.
_MODE_ALIASES = {
    "orthogonal": "orthogonal",
    "covariance": "covariance",
    "covariance_regularized": "covariance",
}


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionTarget:
    """A single editable weight tensor discovered on a model.

    Attributes:
        name: Dotted module path (used as the layer id in edit specs / traces).
        weight: The live parameter tensor (never copied).
        device: ``weight.device`` cached to avoid repeated attribute access.
    """

    name: str
    weight: Tensor
    device: torch.device


def cross_attention_targets(
    model: Any,
    *,
    include: Sequence[str] = ("to_k", "to_v"),
    exclude: Sequence[str] = ("to_q", "to_out"),
) -> List[ProjectionTarget]:
    """Return non-tied projection weights whose module name matches ``include``.

    Walk ``model.named_modules()`` in registration (depth) order and keep 2D or
    3D floating-point weight tensors.  Tied parameters are de-duplicated so a
    shared weight is cloned/edited/restored exactly once.  ``to_q`` / ``to_out``
    are excluded by default because the concept lives in the key/value branch.
    """
    targets: List[ProjectionTarget] = []
    seen: set = set()
    for module_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor) or weight.ndim not in (2, 3):
            continue
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            continue
        leaf = module_name.rsplit(".", 1)[-1]
        if include and not any(token in module_name or token == leaf for token in include):
            continue
        if any(token in module_name or token == leaf for token in exclude):
            continue
        pointer = (weight.data_ptr(), tuple(weight.shape))
        if pointer in seen:
            continue
        seen.add(pointer)
        targets.append(ProjectionTarget(name=module_name, weight=weight, device=weight.device))
    return targets


def resolve_targets(
    targets: Sequence[ProjectionTarget], names: Sequence[str]
) -> List[ProjectionTarget]:
    """Select ``targets`` by name, preserving the requested order."""
    by_name = {target.name: target for target in targets}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"Unknown target layer(s): {missing[:5]}.")
    return [by_name[name] for name in names]


# -----------------------------------------------------------------------------
# Heretic-style alpha kernel
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class InterpolationKernel:
    """Parameters of the per-layer scaling kernel.

    Attributes:
        alpha_max: Peak scaling applied at the kernel centre.
        peak_position: Centre as a fraction of depth in ``[0, 1]`` (``0`` = the
            shallowest eligible layer, ``1`` = the deepest).
        alpha_min: Floor scaling approached far from the centre.
        falloff_distance: Decay length in *layer-index* units; it also shifts the
            effective peak toward shallower layers when ``peak_position < 1``,
            matching the Heretic convention so deep layers are not over-edited.
    """

    alpha_max: float
    peak_position: float = 0.5
    alpha_min: float = 0.0
    falloff_distance: float = 0.25


def alpha_kernel(num_layers: int, kernel: InterpolationKernel) -> np.ndarray:
    """Return the ``(num_layers,)`` scaling vector for a Heretic-style kernel."""
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    if not math.isfinite(kernel.alpha_max) or kernel.alpha_max < 0:
        raise ValueError("alpha_max must be finite and nonnegative.")
    if not math.isfinite(kernel.alpha_min) or kernel.alpha_min < 0:
        raise ValueError("alpha_min must be finite and nonnegative.")
    if kernel.alpha_max < kernel.alpha_min:
        raise ValueError("alpha_max must be >= alpha_min.")
    if not 0.0 <= kernel.peak_position <= 1.0:
        raise ValueError("peak_position must be in [0, 1].")
    if not math.isfinite(kernel.falloff_distance) or kernel.falloff_distance <= 0:
        raise ValueError("falloff_distance must be finite and positive.")
    if num_layers == 1:
        return np.array([kernel.alpha_max], dtype=np.float64)
    # Centre in index space: peak_position=0 is the shallowest layer, 1 the
    # deepest.  The Gaussian-like falloff scales with layer-index distance.
    centre = kernel.peak_position * (num_layers - 1)
    indices = np.arange(num_layers, dtype=np.float64)
    decay = np.exp(-np.abs(indices - centre) / kernel.falloff_distance)
    return kernel.alpha_min + (kernel.alpha_max - kernel.alpha_min) * decay


def resolve_alphas(
    num_layers: int,
    kernel: InterpolationKernel,
    *,
    layer_mode: str = "range",
    mask: Optional[Sequence[bool]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> np.ndarray:
    """Return the full per-layer alpha vector, zeroing unselected layers.

    ``layer_mode="mask"`` uses ``mask`` (one bool per eligible layer);
    ``layer_mode="range"`` selects ``[start:end]``.  The kernel is always
    evaluated over the *full* eligible depth so ``peak_position`` is relative to
    the whole model, then unselected entries are zeroed.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    alphas = alpha_kernel(num_layers, kernel).copy()
    if layer_mode == "mask":
        if mask is None:
            raise ValueError("mask mode requires a boolean mask.")
        enabled = np.asarray(list(mask), dtype=bool)
        if enabled.shape != (num_layers,):
            raise ValueError("mask must provide one boolean per eligible layer.")
    elif layer_mode == "range":
        lo = 0 if start is None else int(start)
        hi = num_layers if end is None else int(end)
        if not 0 <= lo <= hi <= num_layers:
            raise ValueError("range must satisfy 0 <= start <= end <= num_layers.")
        enabled = np.zeros(num_layers, dtype=bool)
        enabled[lo:hi] = True
    else:
        raise ValueError("layer_mode must be 'mask' or 'range'.")
    return np.where(enabled, alphas, 0.0)


# -----------------------------------------------------------------------------
# Application / restore
# -----------------------------------------------------------------------------


def _normalize_mode(mode: str) -> str:
    try:
        return _MODE_ALIASES[mode]
    except KeyError:
        raise ValueError(f"mode must be one of {sorted(_MODE_ALIASES)}.") from None


def apply_alpha(
    weight: Tensor,
    directions: Tensor,
    alpha: float,
    *,
    mode: str = "orthogonal",
    side: str = "input",
    neutral: Optional[Tensor] = None,
    covariance: Optional[Tensor] = None,
    regularization: float = 1e-4,
    compute_dtype: torch.dtype = torch.float64,
    chunk_size: int = 256,
) -> Tensor:
    """In-place ``W <- lerp(W, P W, alpha)``; alpha 0/1 take exact shortcuts.

    Returns the same tensor object (storage, strides and ties preserved, as in
    :func:`heretic_dit.core.projector.project_weights_`).  ``alpha == 0`` is a
    no-op and ``alpha == 1`` copies the projection directly - both avoid the
    ``W + 1 * (P - W)`` cancellation a naive interpolation would incur.
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")
    if alpha == 0.0:
        return weight
    projected = project_weights(
        weight,
        directions,
        mode=_normalize_mode(mode),
        side=side,
        neutral=neutral,
        covariance=covariance,
        regularization=regularization,
        compute_dtype=compute_dtype,
        chunk_size=chunk_size,
    )
    with torch.no_grad():
        if alpha == 1.0:
            weight.copy_(projected)
        else:
            weight.lerp_(projected, alpha)
    del projected
    return weight


@dataclass(frozen=True)
class LayerEdit:
    """A fully-resolved edit for one weight tensor.

    Attributes:
        name: Layer id (for traces).
        weight: Live parameter to mutate.
        alpha: Interpolation factor in ``[0, inf)``.
        directions: Projection directions consumed by :mod:`heretic_dit.core.projector`.
        mode: ``"orthogonal"`` / ``"covariance"`` / ``"covariance_regularized"``.
        side: ``"input"`` (``W P``) or ``"output"`` (``P W``).
        neutral: Calibration activations for covariance mode.
        covariance: Alternative precomputed covariance.
        regularization: Ridge for covariance mode.
    """

    name: str
    weight: Tensor
    alpha: float
    directions: Tensor
    mode: str = "orthogonal"
    side: str = "input"
    neutral: Optional[Tensor] = None
    covariance: Optional[Tensor] = None
    regularization: float = 1e-4


def _apply_one(edit: LayerEdit) -> None:
    apply_alpha(
        edit.weight,
        edit.directions,
        edit.alpha,
        mode=edit.mode,
        side=edit.side,
        neutral=edit.neutral,
        covariance=edit.covariance,
        regularization=edit.regularization,
    )


def apply_edits(model: Any, edits: Sequence[LayerEdit]) -> List[Tuple[Tensor, Tensor]]:
    """Apply ``edits`` and return ``(weight, snapshot)`` pairs for restoration.

    Every edited weight is snapshotted *before* any mutation begins, so a
    projector that raises partway through leaves the model bit-exactly as it was
    (snapshots restore in an ``except`` handler) rather than half-edited.
    """
    del model  # edits carry their own weight references.
    snapshots: List[Tuple[Tensor, Tensor]] = [
        (edit.weight, edit.weight.detach().clone()) for edit in edits if edit.alpha != 0.0
    ]
    try:
        for edit in edits:
            if edit.alpha != 0.0:
                _apply_one(edit)
    except BaseException:
        restore_edits(snapshots)
        raise
    return snapshots


def restore_edits(snapshots: Sequence[Tuple[Tensor, Tensor]]) -> None:
    """Copy every snapshot back into its weight, preserving storage identity."""
    with torch.no_grad():
        for weight, snapshot in snapshots:
            weight.copy_(snapshot)


@contextmanager
def applied_edit(model: Any, edits: Sequence[LayerEdit]) -> Iterator[None]:
    """Context manager applying per-layer edits and restoring them unconditionally.

    Each edited weight is cloned immediately before mutation and copied back in a
    ``finally`` block, so restoration is bit-exact even when the body raises -
    a crashed trial can never leave the model corrupted.
    """
    snapshots = apply_edits(model, edits)
    try:
        yield
    finally:
        restore_edits(snapshots)