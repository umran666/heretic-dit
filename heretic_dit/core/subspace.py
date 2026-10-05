"""Deterministic concept-subspace extraction from collected activations.

Inputs may be (samples, dim), (batch, sequence, dim), or
(batch, timesteps, sequence, dim). Without timestep weights, every position
is a sample. With weights, the temporal axis is collapsed using the literal
sum_t weights[t] * X[:, t], then batch and sequence are flattened. This is
trajectory aggregation, not sample weighting; weights are not normalized.

All outputs are detached CPU tensors in compute_dtype (float64 by default).
Autocast is disabled during all arithmetic. SVD/eigh signs use a positive
largest-magnitude pivot, whereas mean differences preserve target-minus-neutral
orientation. Repeatability applies to a fixed software/hardware stack; repeated
eigenvalues can admit different bases across stacks despite sign canonicalization.

PCA/SVD center their inputs by default. compute_covariance defaults to uncentered
second moments, matching X.T @ X / N. Its shrinkage estimators and explicit ridge
are distinct: numeric shrinkage is a coefficient toward a scaled identity,
not the ridge lambda. Pass the result to projector.project_weights(covariance=...).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal, Optional, Tuple, Union

import torch
from torch import Tensor

Shrinkage = Union[Literal["ledoit_wolf", "diagonal"], float]

__all__ = [
    "Shrinkage", "aggregate_activations", "mean_difference", "contrastive_pca",
    "subspace_svd", "compute_covariance",
]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _validate_dtype(compute_dtype: torch.dtype) -> None:
    if compute_dtype not in (torch.float32, torch.float64):
        raise TypeError("compute_dtype must be torch.float32 or torch.float64.")


def _require_finite(tensor: Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity (possibly after casting).")


def _validate_real(value: float, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}.")
    return value


def _validate_rank(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _relative_tolerance(rcond: Optional[float], shape: Tuple[int, ...], dtype: torch.dtype) -> float:
    if rcond is None:
        return max(shape) * torch.finfo(dtype).eps
    value = _validate_real(rcond, "rcond")
    if value >= 1:
        raise ValueError("rcond must be in [0, 1).")
    return value


def _validate_center(center: bool) -> None:
    if not isinstance(center, bool):
        raise TypeError("center must be a bool.")


def _canonicalize(basis: Tensor) -> Tensor:
    if basis.shape[1] == 0:
        return basis.contiguous()
    pivots = basis.abs().argmax(dim=0)
    signs = basis.gather(0, pivots.unsqueeze(0)).sign()
    return (basis * signs).contiguous()


def _scaled_samples(samples: Tensor, scale: Tensor, center: bool) -> Tensor:
    scaled = samples / scale if float(scale) > 0 else samples
    if center:
        # Shift first so constant samples center to exact zero without sum drift.
        shifted = scaled - scaled[:1]
        scaled = shifted - shifted.mean(dim=0, keepdim=True)
    return scaled


def _normalized_samples(samples: Tensor, center: bool) -> Tuple[Tensor, Tensor]:
    """Center before scaling when subtraction is representable, preserving offsets."""
    if center:
        shifted = samples - samples[:1]
        if bool(torch.isfinite(shifted).all()):
            scale = shifted.abs().max()
            scaled = shifted / scale if float(scale) > 0 else shifted
            return scaled - scaled.mean(dim=0, keepdim=True), scale
    scale = samples.abs().max()
    # Opposite extreme values can overflow an original-unit shift; scale first.
    return _scaled_samples(samples, scale, center), scale


def _second_moment(samples: Tensor) -> Tensor:
    moment = (samples.T @ samples) / samples.shape[0]
    _require_finite(moment, "second moment")
    return moment * 0.5 + moment.T * 0.5


@torch.no_grad()
def aggregate_activations(
    activations: Tensor,
    *,
    timestep_weights: Optional[Tensor] = None,
    compute_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Return owned (num_samples, dim) CPU activations, optionally summing time.

    timestep_weights must be a finite, nonnegative (timesteps,) tensor with at
    least one positive entry, and is accepted only for 4D activations. Zero
    entries can select a semantic timestep window. Normalize weights explicitly
    before passing them if a weighted average rather than a sum is desired.
    For 3D inputs the middle axis is sequence, never implicitly timesteps.
    Inputs, including noncontiguous tensors and tensors requiring grad, are
    preserved. Empty dimensions, unsupported dtypes and nonfinite data raise.
    """
    _validate_dtype(compute_dtype)
    if not isinstance(activations, Tensor):
        raise TypeError("activations must be a torch.Tensor.")
    if activations.layout != torch.strided or activations.device.type == "meta":
        raise ValueError("activations must be a materialized, strided tensor.")
    if activations.dtype not in _FLOAT_DTYPES:
        raise TypeError("activations must have a real floating-point dtype.")
    if activations.ndim not in (2, 3, 4) or any(size == 0 for size in activations.shape):
        raise ValueError("activations must have nonempty shape (N,d), (B,S,d), or (B,T,S,d).")
    with torch.autocast(device_type="cpu", enabled=False):
        samples = activations.detach().to(device="cpu", dtype=compute_dtype, copy=True)
        _require_finite(samples, "activations")
        if timestep_weights is not None:
            if activations.ndim != 4:
                raise ValueError("timestep_weights require 4D (B,T,S,d) activations.")
            if not isinstance(timestep_weights, Tensor):
                raise TypeError("timestep_weights must be a torch.Tensor.")
            if timestep_weights.layout != torch.strided or timestep_weights.device.type == "meta":
                raise ValueError("timestep_weights must be a materialized, strided tensor.")
            if timestep_weights.dtype not in _FLOAT_DTYPES:
                raise TypeError("timestep_weights must have a real floating-point dtype.")
            if timestep_weights.ndim != 1 or timestep_weights.shape[0] != samples.shape[1]:
                raise ValueError("timestep_weights must have shape (timesteps,).")
            weights = timestep_weights.detach().to(device="cpu", dtype=compute_dtype)
            _require_finite(weights, "timestep_weights")
            if bool((weights < 0).any()) or not bool((weights > 0).any()):
                raise ValueError("timestep_weights must be nonnegative with a positive entry.")
            samples = torch.einsum("btsd,t->bsd", samples, weights)
            _require_finite(samples, "aggregated activations")
        return samples.reshape(-1, samples.shape[-1]).contiguous()


@torch.no_grad()
def mean_difference(
    target_activations: Tensor,
    neutral_activations: Tensor,
    *,
    timestep_weights: Optional[Tensor] = None,
    compute_dtype: torch.dtype = torch.float64,
    rcond: Optional[float] = None,
) -> Tensor:
    """Return the normalized target-minus-neutral mean direction as (dim, 1).

    Cohorts may have different sample counts but must share the feature width.
    Timestep aggregation follows aggregate_activations. Scaling both cohorts
    together avoids overflow without changing direction. A shared reference is
    subtracted first when representable, preserving small differences behind
    common offsets. A difference whose norm, relative to the largest shifted
    activation magnitude, is <= rcond raises ValueError; the default cutoff is
    dim * finfo(compute_dtype).eps. If shifting overflows, use raw scaled means.
    The sign preserves the mean difference; it is not an arbitrary SVD sign.
    """
    with torch.autocast(device_type="cpu", enabled=False):
        target = aggregate_activations(target_activations, timestep_weights=timestep_weights, compute_dtype=compute_dtype)
        neutral = aggregate_activations(neutral_activations, timestep_weights=timestep_weights, compute_dtype=compute_dtype)
        if target.shape[1] != neutral.shape[1]:
            raise ValueError("target and neutral must have matching feature dimensions.")
        cutoff = _relative_tolerance(rcond, (target.shape[1],), compute_dtype)
        target_shifted = target - target[:1]
        neutral_shifted = neutral - target[:1]
        if bool(torch.isfinite(target_shifted).all()) and bool(torch.isfinite(neutral_shifted).all()):
            target, neutral = target_shifted, neutral_shifted
        scale = torch.maximum(target.abs().max(), neutral.abs().max())
        difference = _scaled_samples(target, scale, False).mean(dim=0) - _scaled_samples(neutral, scale, False).mean(dim=0)
        norm = torch.linalg.vector_norm(difference)
        if float(norm) <= cutoff:
            raise ValueError("Target and neutral means have a numerically zero difference.")
        direction = (difference / norm).unsqueeze(1).contiguous()
        _require_finite(direction, "mean direction")
        return direction


@torch.no_grad()
def contrastive_pca(
    target_activations: Tensor,
    neutral_activations: Tensor,
    k: int = 1,
    *,
    alpha: float = 1.0,
    center: bool = True,
    timestep_weights: Optional[Tensor] = None,
    compute_dtype: torch.dtype = torch.float64,
    rcond: Optional[float] = None,
) -> Tensor:
    """Return up to k positive principal directions of C_target - alpha*C_neutral.

    Each covariance uses its own 1/N normalization and, by default, its own
    centered samples. Eigenspaces are ordered by descending algebraic eigenvalue,
    never absolute eigenvalue. Negative and numerical-null directions are
    excluded: return (dim, r) with r <= k, or (dim, 0) if no positive contrast
    exists. k must be in [1, dim]; alpha must be finite and nonnegative.
    rcond defaults to dim * finfo(compute_dtype).eps, relative to the larger of
    the contrastive eigenvalue scale and constituent covariance infinity norms.
    This excludes roundoff-only contrast between identical distributions.
    Signs use positive magnitude pivots. alpha must fit in compute_dtype.
    """
    _validate_rank(k, "k")
    alpha = _validate_real(alpha, "alpha")
    _validate_center(center)
    with torch.autocast(device_type="cpu", enabled=False):
        target = aggregate_activations(target_activations, timestep_weights=timestep_weights, compute_dtype=compute_dtype)
        neutral = aggregate_activations(neutral_activations, timestep_weights=timestep_weights, compute_dtype=compute_dtype)
        if alpha > torch.finfo(compute_dtype).max:
            raise ValueError("alpha is not representable in compute_dtype.")
        d = target.shape[1]
        if neutral.shape[1] != d:
            raise ValueError("target and neutral must have matching feature dimensions.")
        if k > d:
            raise ValueError("k cannot exceed the feature dimension.")
        cutoff = _relative_tolerance(rcond, (d,), compute_dtype)
        target_scaled, target_scale = _normalized_samples(target, center)
        neutral_scaled, neutral_scale = _normalized_samples(neutral, center)
        scale = torch.maximum(target_scale, neutral_scale)
        if float(scale) > 0:
            target_scaled = target_scaled * (target_scale / scale)
            neutral_scaled = neutral_scaled * (neutral_scale / scale)
        target_cov = _second_moment(target_scaled)
        neutral_cov = _second_moment(neutral_scaled)
        # Overall scaling leaves eigenvectors unchanged and bounds large alpha.
        alpha_scale = max(1.0, alpha)
        target_cov.div_(alpha_scale)
        neutral_cov.mul_(alpha / alpha_scale)
        background_scale = torch.maximum(
            torch.linalg.matrix_norm(target_cov, ord=float("inf")),
            torch.linalg.matrix_norm(neutral_cov, ord=float("inf")),
        )
        contrast = target_cov - neutral_cov
        contrast = contrast * 0.5 + contrast.T * 0.5
        eigenvalues, eigenvectors = torch.linalg.eigh(contrast)
        reference = torch.maximum(eigenvalues.abs().max(), background_scale)
        keep = eigenvalues > cutoff * reference
        indices = keep.nonzero(as_tuple=True)[0].flip(0)[:k]
        return _canonicalize(eigenvectors[:, indices])


@torch.no_grad()
def subspace_svd(
    activations: Tensor,
    energy_threshold: float = 0.9,
    max_rank: int = 8,
    *,
    center: bool = True,
    timestep_weights: Optional[Tensor] = None,
    compute_dtype: torch.dtype = torch.float64,
    rcond: Optional[float] = None,
) -> Tensor:
    """Return an orthonormal (dim, rank) basis selected by squared-SVD energy.

    Select the smallest numerical rank reaching energy_threshold in (0, 1],
    then cap it at max_rank. The cap can prevent reaching the requested energy.
    Centering is enabled by default; disable it to extract second-moment energy.
    Constant centered data returns (dim, 0), never arbitrary null directions.
    Numerical rank uses rcond relative to the largest singular value, defaulting
    to max(num_samples, dim) * finfo(compute_dtype).eps. Energy is measured within
    this numerical span. SVD signs use positive largest-magnitude pivots.
    """
    energy_threshold = _validate_real(energy_threshold, "energy_threshold")
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1].")
    _validate_rank(max_rank, "max_rank")
    _validate_center(center)
    with torch.autocast(device_type="cpu", enabled=False):
        samples = aggregate_activations(activations, timestep_weights=timestep_weights, compute_dtype=compute_dtype)
        cutoff = _relative_tolerance(rcond, tuple(samples.shape), compute_dtype)
        scaled, _ = _normalized_samples(samples, center)
        _, singular_values, vh = torch.linalg.svd(scaled, full_matrices=False)
        if float(singular_values[0]) == 0:
            return samples.new_empty((samples.shape[1], 0))
        numerical_rank = int((singular_values > cutoff * singular_values[0]).sum())
        if numerical_rank == 0:
            return samples.new_empty((samples.shape[1], 0))
        energy = (singular_values[:numerical_rank] / singular_values[0]).square()
        cumulative = energy.cumsum(dim=0) / energy.sum()
        required = int(torch.searchsorted(cumulative, energy_threshold)) + 1
        rank = min(required, numerical_rank, max_rank)
        return _canonicalize(vh[:rank].T)


def _ledoit_wolf_coefficient(samples: Tensor, empirical: Tensor) -> float:
    """LW isotropic shrinkage using O(N*d + d**2) storage, without sklearn.

    Formula matches sklearn.covariance.ledoit_wolf with assume_centered=True
    on the supplied samples (which may already have been centered).
    Reference: sklearn/covariance/_shrunk_covariance.py, ledoit_wolf_shrinkage.
    """
    d = empirical.shape[0]
    mu = empirical.diagonal().mean()
    deviation = empirical.clone()
    deviation.diagonal().sub_(mu)
    delta = deviation.square().sum()
    if float(delta) == 0:
        return 0.0
    fourth_moment = samples.square().sum(dim=1).square().mean()
    beta = (fourth_moment - empirical.square().sum()) / samples.shape[0]
    return float((beta / delta).clamp(0, 1))


@torch.no_grad()
def compute_covariance(
    activations: Tensor,
    shrinkage: Shrinkage = "ledoit_wolf",
    *,
    regularization: float = 1e-4,
    center: bool = False,
    timestep_weights: Optional[Tensor] = None,
    compute_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Estimate a symmetric (dim, dim) covariance/second moment plus ridge.

    Let S = X.T @ X / N after optional centering and timestep aggregation.
    All modes add regularization * I in the original activation units:

    * float rho in [0,1]: (1-rho)*S + rho*trace(S)/dim*I + lambda*I;
    * 'ledoit_wolf': the same expression with a data-estimated rho;
    * 'diagonal': diag(diag(S)) + lambda*I.

    shrinkage=0.0 gives exactly the requested S + lambda*I. Positive ridge makes
    the covariance positive definite, subject to floating-point resolution.
    Both shrinkage and ridge may be zero; singular PSD results are then allowed.
    Centering defaults to False to preserve the uncentered calibration formula.
    Computation is CPU float32/64 with scaling before fourth moments, and inputs
    are preserved. Nonrepresentable covariance values raise ValueError.
    Pass the result as projector covariance=, avoiding a second ridge addition.
    """
    regularization = _validate_real(regularization, "regularization")
    _validate_center(center)
    if isinstance(shrinkage, str):
        if shrinkage not in ("ledoit_wolf", "diagonal"):
            raise ValueError("shrinkage must be 'ledoit_wolf', 'diagonal', or a coefficient in [0,1].")
    else:
        shrinkage = _validate_real(shrinkage, "shrinkage")
        if shrinkage > 1:
            raise ValueError("numeric shrinkage must be in [0,1].")
    with torch.autocast(device_type="cpu", enabled=False):
        samples = aggregate_activations(activations, timestep_weights=timestep_weights, compute_dtype=compute_dtype)
        if regularization > torch.finfo(compute_dtype).max:
            raise ValueError("regularization is not representable in compute_dtype.")
        scaled, scale = _normalized_samples(samples, center)
        empirical = _second_moment(scaled)
        if shrinkage == "diagonal":
            covariance = torch.diag(empirical.diagonal())
        else:
            rho = _ledoit_wolf_coefficient(scaled, empirical) if shrinkage == "ledoit_wolf" else float(shrinkage)
            covariance = empirical * (1 - rho)
            covariance.diagonal().add_(rho * empirical.diagonal().mean())
        # Two multiplications avoid overflowing scale**2 before cancellation.
        covariance = (covariance * scale) * scale
        covariance.diagonal().add_(regularization)
        _require_finite(covariance, "regularized covariance")
        return covariance.contiguous()
