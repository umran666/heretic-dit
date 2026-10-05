"""Closed-form projections for dense and batched expert weights.

Directions are columns of a shared (d, k) matrix, or a single (d,) vector.
Both modes use P = I - A V.T, with V an orthonormal basis:

* orthogonal: A = V;
* covariance: A = Sigma^-1 V (V.T Sigma^-1 V)^-1.

The covariance projector is generally oblique: V.T P = 0, but P V need
not vanish. Input and output projection mean W P and P W, respectively.
This implements the stated covariance formula, not the full LEACE estimator.

SVD and covariance solves are pinned to CPU, without randomized algorithms.
SVD signs are canonicalized; rotations within repeated singular subspaces
cancel in the projector. Repeated execution is reproducible on a fixed
software/hardware stack, not necessarily bitwise across different stacks.
CUDA weight multiplication follows the caller's deterministic-algorithm,
CUBLAS_WORKSPACE_CONFIG, and TF32 settings; no global settings are changed.

This module performs parameter surgery, without constructing autograd graphs.
In-place updates preserve storage, including tied parameters. Chunking bounds
weight temporaries and batches every expert together; no dense P is allocated.
Covariance mode still requires O(d**2) CPU storage and a dense factorization.
"""

from __future__ import annotations

import math
from typing import Iterator, Literal, Optional, Tuple

import torch
from torch import Tensor

ProjectionMode = Literal["orthogonal", "covariance"]
ProjectionSide = Literal["input", "output"]

__all__ = ["ProjectionMode", "ProjectionSide", "project_weights", "project_weights_"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _validate_tensor(tensor: Tensor, name: str, dimensions: Tuple[int, ...]) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError(f"{name} must be a materialized, strided tensor.")
    if tensor.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have a real floating-point dtype.")
    if tensor.ndim not in dimensions:
        raise ValueError(f"{name} must have one of these ranks: {dimensions}.")


def _require_finite(tensor: Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity (possibly after casting).")


def _validate_options(
    mode: ProjectionMode,
    side: ProjectionSide,
    regularization: float,
    compute_dtype: torch.dtype,
    rcond: Optional[float],
    chunk_size: int,
) -> None:
    if mode not in ("orthogonal", "covariance"):
        raise ValueError("mode must be 'orthogonal' or 'covariance'.")
    if side not in ("input", "output"):
        raise ValueError("side must be 'input' or 'output'.")
    if compute_dtype not in (torch.float32, torch.float64):
        raise TypeError("compute_dtype must be torch.float32 or torch.float64.")
    if not math.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if rcond is not None and (not math.isfinite(rcond) or not 0 <= rcond < 1):
        raise ValueError("rcond must be finite and in [0, 1).")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")


def _basis(directions: Tensor, d: int, dtype: torch.dtype, rcond: Optional[float]) -> Tensor:
    _validate_tensor(directions, "directions", (1, 2))
    if directions.shape[0] != d:
        raise ValueError(f"directions must have {d} rows for the selected side.")
    matrix = directions.detach().to(device="cpu", dtype=dtype)
    if matrix.ndim == 1:
        matrix = matrix.unsqueeze(1)
    _require_finite(matrix, "directions")
    if matrix.shape[1] == 0:
        return matrix.clone()
    scale = matrix.abs().max()
    if float(scale) == 0:
        return matrix[:, :0].clone()
    # Scaling avoids overflow without changing the span or relative rank.
    u, singular_values, _ = torch.linalg.svd(matrix / scale, full_matrices=False)
    cutoff = rcond if rcond is not None else max(matrix.shape) * torch.finfo(dtype).eps
    rank = int((singular_values > cutoff * singular_values[0]).sum())
    basis = u[:, :rank].contiguous()
    pivots = basis.abs().argmax(dim=0)
    signs = basis.gather(0, pivots.unsqueeze(0)).sign()
    return basis * signs


def _calibration_covariance(
    neutral: Optional[Tensor],
    covariance: Optional[Tensor],
    d: int,
    regularization: float,
    dtype: torch.dtype,
    chunk_size: int,
) -> Tensor:
    if (neutral is None) == (covariance is None):
        raise ValueError("Covariance mode requires exactly one of neutral or covariance.")
    if covariance is not None:
        _validate_tensor(covariance, "covariance", (2,))
        if covariance.shape != (d, d):
            raise ValueError(f"covariance must have shape ({d}, {d}).")
        sigma = covariance.detach().to(device="cpu", dtype=dtype).clone()
    else:
        if neutral is None:
            raise ValueError("neutral calibration activations are required.")
        _validate_tensor(neutral, "neutral", (2,))
        if neutral.shape[1] != d or neutral.shape[0] == 0:
            raise ValueError(f"neutral must have shape (num_samples > 0, {d}).")
        sigma = torch.zeros((d, d), dtype=dtype, device="cpu")
        for start in range(0, neutral.shape[0], chunk_size):
            block = neutral[start : start + chunk_size].detach().to(device="cpu", dtype=dtype)
            _require_finite(block, "neutral")
            sigma.addmm_(block.T, block)
        sigma.diagonal().add_(regularization)
    _require_finite(sigma, "regularized covariance")
    # Scaling cancels in A; do it before symmetrizing to avoid underflow.
    scale = sigma.abs().max()
    if float(scale) > 0:
        sigma = sigma / scale
    tolerance = 16 * d * torch.finfo(dtype).eps
    if covariance is not None and float((sigma - sigma.T).abs().max()) > tolerance:
        raise ValueError("covariance must be symmetric.")
    return sigma * 0.5 + sigma.T * 0.5


def _solve_psd(matrix: Tensor, rhs: Tensor, rcond: Optional[float]) -> Tensor:
    """Cholesky solve, with a checked spectral pseudoinverse for singular PSD input."""
    _require_finite(matrix, "covariance solve matrix")
    _require_finite(rhs, "covariance solve right-hand side")
    factor, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
    if int(info) == 0:
        result = torch.cholesky_solve(rhs, factor)
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        eps = torch.finfo(matrix.dtype).eps
        scale = eigenvalues.abs().max()
        if float(eigenvalues.min()) < -16 * matrix.shape[0] * eps * float(scale):
            raise ValueError("covariance and its direction Gram matrix must be positive semidefinite.")
        cutoff = rcond if rcond is not None else matrix.shape[0] * eps
        keep = eigenvalues > cutoff * scale
        inverse = torch.zeros_like(eigenvalues)
        inverse[keep] = eigenvalues[keep].reciprocal()
        result = eigenvectors @ (inverse.unsqueeze(1) * (eigenvectors.T @ rhs))
    _require_finite(result, "covariance solve")
    return result


def _projection_factors(
    directions: Tensor,
    d: int,
    mode: ProjectionMode,
    neutral: Optional[Tensor],
    covariance: Optional[Tensor],
    regularization: float,
    dtype: torch.dtype,
    rcond: Optional[float],
    chunk_size: int,
) -> Tuple[Tensor, Tensor]:
    # Explicitly disable outer CPU autocast for calibration and factorization.
    with torch.autocast(device_type="cpu", enabled=False):
        basis = _basis(directions, d, dtype, rcond)
        if mode == "orthogonal":
            if neutral is not None or covariance is not None:
                raise ValueError("neutral and covariance apply only to covariance mode.")
            return basis, basis
        sigma = _calibration_covariance(neutral, covariance, d, regularization, dtype, chunk_size)
        if basis.shape[1] == 0:
            # Validate PSD even when the requested subspace is empty.
            _solve_psd(sigma, basis, rcond)
            return basis, basis
        solved = _solve_psd(sigma, basis, rcond)
        gram = basis.T @ solved
        gram = gram * 0.5 + gram.T * 0.5
        left = _solve_psd(gram, solved.T, rcond).T.contiguous()
        identity = torch.eye(basis.shape[1], dtype=dtype)
        tolerance = max(1e-10, 32 * basis.shape[1] * torch.finfo(dtype).eps)
        if not torch.allclose(basis.T @ left, identity, rtol=tolerance, atol=tolerance):
            raise ValueError(
                "Covariance cannot support the complete target subspace at this precision. "
                "Use positive regularization, float64, or a less singular calibration covariance."
            )
        return left, basis


def _check_nonoverlapping(weight: Tensor) -> None:
    """Accept standard transposes/slices; reject overlapping or ambiguous strides."""
    span = 1
    for stride, size in sorted(zip(weight.stride(), weight.shape)):
        if size > 1:
            if stride < span:
                raise ValueError("In-place projection requires nonoverlapping weight storage.")
            span += (size - 1) * stride


def _weight_blocks(weight: Tensor, side: ProjectionSide, chunk_size: int) -> Iterator[Tensor]:
    # Only the unaffected axis is chunked, so in-place updates are independent.
    axis = -2 if side == "input" else -1
    for start in range(0, weight.shape[axis], chunk_size):
        yield weight.narrow(axis, start, min(chunk_size, weight.shape[axis] - start))


@torch.no_grad()
def _project(
    weight: Tensor,
    directions: Tensor,
    *,
    mode: ProjectionMode,
    side: ProjectionSide,
    neutral: Optional[Tensor],
    covariance: Optional[Tensor],
    regularization: float,
    compute_dtype: torch.dtype,
    rcond: Optional[float],
    chunk_size: int,
    inplace: bool,
) -> Tensor:
    _validate_tensor(weight, "weight", (2, 3))
    if any(size == 0 for size in weight.shape):
        raise ValueError("weight dimensions must be nonzero.")
    _validate_options(mode, side, regularization, compute_dtype, rcond, chunk_size)
    if inplace:
        _check_nonoverlapping(weight)
    d = weight.shape[-1 if side == "input" else -2]
    left, basis = _projection_factors(
        directions, d, mode, neutral, covariance, regularization,
        compute_dtype, rcond, chunk_size,
    )
    # Preflight all source blocks before mutation, without a full-sized mask.
    for block in _weight_blocks(weight, side, chunk_size):
        _require_finite(block.to(dtype=compute_dtype), "weight")
    result = weight if inplace else weight.detach().clone()
    if basis.shape[1] == 0:
        return result
    left = left.to(device=weight.device)
    right = basis.T.contiguous().to(device=weight.device)
    if weight.ndim == 3:
        left = left.unsqueeze(0).expand(weight.shape[0], -1, -1)
        right = right.unsqueeze(0).expand(weight.shape[0], -1, -1)
    multiply = torch.bmm if weight.ndim == 3 else torch.mm
    with torch.autocast(device_type=weight.device.type, enabled=False):
        for block in _weight_blocks(result, side, chunk_size):
            work = block.to(dtype=compute_dtype, copy=True)
            if side == "input":
                first, second = multiply(work, left), right
            else:
                first, second = left, multiply(right, work)
            # Fused accumulation avoids a second full-sized correction buffer.
            if weight.ndim == 3:
                work.baddbmm_(first, second, beta=1, alpha=-1)
            else:
                work.addmm_(first, second, beta=1, alpha=-1)
            _require_finite(work, "projected weight")
            native = work.to(dtype=weight.dtype)
            _require_finite(native, "projected weight in native dtype")
            block.copy_(native)
            del work, native, first, second
    return result


def project_weights(
    weight: Tensor,
    directions: Tensor,
    *,
    mode: ProjectionMode = "orthogonal",
    side: ProjectionSide = "input",
    neutral: Optional[Tensor] = None,
    covariance: Optional[Tensor] = None,
    regularization: float = 1e-4,
    compute_dtype: torch.dtype = torch.float64,
    rcond: Optional[float] = None,
    chunk_size: int = 256,
) -> Tensor:
    """Return a detached projected copy, preserving weight shape, device and dtype.

    Args:
        weight: Dense (out, in) or batched expert (experts, out, in) weights.
        directions: Shared (d, k) direction columns, or a single (d,) vector.
            d is the input or output width selected by side. Dependent columns
            are reduced to their numerical span; an empty/zero span is a no-op.
        mode: 'orthogonal' for I - V V.T; 'covariance' for the formula above.
        side: 'input' computes W P; 'output' computes P W.
        neutral: In covariance mode, uncentered (samples, d) activations.
            Sigma = neutral.T @ neutral + regularization * I, with no averaging.
        covariance: Alternatively, an already regularized, symmetric PSD Sigma.
            Used as supplied; regularization is not added a second time.
            Supply exactly one of neutral or covariance in covariance mode.
        regularization: Nonnegative ridge applied only when neutral is supplied.
        compute_dtype: float64 by default, or float32. All projection arithmetic
            uses this dtype, with autocast disabled, before the final cast.
        rcond: Relative cutoff for direction rank and singular PSD pseudoinverses.
            Defaults to max(matrix.shape) * finfo(compute_dtype).eps per matrix.
            Cholesky is used directly for positive definite matrices.
        chunk_size: Rows per input-projection block, columns per output-projection
            block, and samples per calibration block. All experts remain batched.

    Raises:
        TypeError: Unsupported tensor or computation dtype.
        ValueError: Invalid dimensions/options, nonfinite values, indefinite
            covariance, or a covariance unable to erase the entire target span.

    Native low-precision rounding can leave a small projection residual.
    This is a weight-only operation; output biases are not modified.
    """
    return _project(
        weight, directions, mode=mode, side=side, neutral=neutral,
        covariance=covariance, regularization=regularization,
        compute_dtype=compute_dtype, rcond=rcond, chunk_size=chunk_size, inplace=False,
    )


def project_weights_(
    weight: Tensor,
    directions: Tensor,
    *,
    mode: ProjectionMode = "orthogonal",
    side: ProjectionSide = "input",
    neutral: Optional[Tensor] = None,
    covariance: Optional[Tensor] = None,
    regularization: float = 1e-4,
    compute_dtype: torch.dtype = torch.float64,
    rcond: Optional[float] = None,
    chunk_size: int = 256,
) -> Tensor:
    """Project weight through chunked copy_ updates and return that same object.

    Arguments and mathematics match project_weights. Preserves data_ptr(),
    storage, strides, requires_grad, and ties; never replaces Parameter.data.
    Supports standard nonoverlapping transposes and slices. Expanded/overlapping
    tensors and ambiguous strides are rejected before any mutation.

    Workspace scales with one chunk across all experts, not a full float32/64
    weight copy. Input validation completes before writes. Updates are not
    transactional: a runtime/OOM or arithmetic overflow in a later chunk can
    leave earlier chunks updated. Use project_weights for an atomic preparation
    of a separate result before committing it to model storage.
    """
    return _project(
        weight, directions, mode=mode, side=side, neutral=neutral,
        covariance=covariance, regularization=regularization,
        compute_dtype=compute_dtype, rcond=rcond, chunk_size=chunk_size, inplace=True,
    )


def _self_test() -> None:
    """Small deterministic assertions, runnable with python -m ...projector."""
    generator = torch.Generator(device="cpu").manual_seed(17)
    dtype = torch.float64
    for shape in ((7, 5), (128, 7, 5)):
        weight = torch.randn(shape, generator=generator, dtype=dtype)
        for side in ("input", "output"):
            d = shape[-1 if side == "input" else -2]
            directions = torch.randn((d, 2), generator=generator, dtype=dtype)
            basis, _ = torch.linalg.qr(directions, mode="reduced")
            neutral = torch.randn((23, d), generator=generator, dtype=dtype)
            for mode in ("orthogonal", "covariance"):
                options = {"neutral": neutral} if mode == "covariance" else {}
                left = basis
                if mode == "covariance":
                    sigma = neutral.T @ neutral + 1e-4 * torch.eye(d, dtype=dtype)
                    solved = torch.linalg.solve(sigma, basis)
                    left = torch.linalg.solve(basis.T @ solved, solved.T).T
                projector = torch.eye(d, dtype=dtype) - left @ basis.T
                expected = weight @ projector if side == "input" else projector @ weight
                projected = project_weights(weight, directions, mode=mode, side=side, **options)
                torch.testing.assert_close(projected, expected, rtol=1e-10, atol=1e-10)
                repeated = project_weights(weight, directions, mode=mode, side=side, **options)
                assert torch.equal(projected, repeated)
                parameter = torch.nn.Parameter(weight.clone())
                tied = parameter.detach()
                pointer = parameter.data_ptr()
                returned = project_weights_(parameter, directions, mode=mode, side=side, chunk_size=3, **options)
                assert returned is parameter and parameter.data_ptr() == pointer
                assert tied.data_ptr() == pointer and parameter.requires_grad
                torch.testing.assert_close(tied, expected, rtol=1e-10, atol=1e-10)
    print("Projector self-checks passed.")


if __name__ == "__main__":
    _self_test()
