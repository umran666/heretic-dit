"""Regression tests for projection algebra and parameter-storage safety."""

import pytest
import torch

from heretic_dit.core.projector import _basis, project_weights, project_weights_


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize("mode", ["orthogonal", "covariance"])
def test_matches_dense_formula_and_projection_identities(batched, side, mode):
    generator = torch.Generator().manual_seed(29)
    shape = (128, 9, 6) if batched else (9, 6)
    weight = torch.randn(shape, dtype=torch.float64, generator=generator)
    original = weight.clone()
    d = shape[-1 if side == "input" else -2]
    directions = torch.randn((d, 3), dtype=torch.float64, generator=generator)
    # QR is an independent reference for the span used by the SVD implementation.
    basis, _ = torch.linalg.qr(directions, mode="reduced")
    sigma = torch.diag(torch.arange(1, d + 1, dtype=torch.float64))
    options = {"covariance": sigma} if mode == "covariance" else {}
    left = basis
    if mode == "covariance":
        solved = torch.linalg.solve(sigma, basis)
        left = torch.linalg.solve(basis.T @ solved, solved.T).T
    projector = torch.eye(d, dtype=torch.float64) - left @ basis.T
    expected = weight @ projector if side == "input" else projector @ weight
    actual = project_weights(weight, directions, mode=mode, side=side, chunk_size=2, **options)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(projector @ projector, projector, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(basis.T @ projector, torch.zeros((3, d), dtype=torch.float64), atol=1e-11, rtol=0)
    assert torch.equal(weight, original)
    assert actual.data_ptr() != weight.data_ptr() and not actual.requires_grad
    repeated = project_weights(actual, directions, mode=mode, side=side, chunk_size=2, **options)
    torch.testing.assert_close(repeated, actual, rtol=1e-11, atol=1e-11)
    if mode == "orthogonal":
        residual = actual @ basis if side == "input" else basis.T @ actual
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-11, rtol=0)


def test_covariance_orientation_is_oblique():
    sigma = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float64)
    projector = project_weights(torch.eye(2, dtype=torch.float64), direction, mode="covariance", covariance=sigma)
    torch.testing.assert_close(direction @ projector, torch.zeros(2, dtype=torch.float64))
    assert not torch.allclose(projector @ direction, torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("side", ["input", "output"])
def test_dtype_and_tied_noncontiguous_parameter_storage(dtype, side):
    base = torch.arange(128 * 6 * 9, dtype=torch.float32).reshape(128, 6, 9).remainder(17).to(dtype)
    parameter = torch.nn.Parameter(base.transpose(-1, -2))
    alias = parameter.detach()
    strides = parameter.stride()
    pointer = parameter.data_ptr()
    d = parameter.shape[-1 if side == "input" else -2]
    direction = torch.zeros(d, dtype=dtype)
    direction[0] = 1
    expected = parameter.detach().clone()
    if side == "input":
        expected[..., 0] = 0
    else:
        expected[..., 0, :] = 0
    returned = project_weights_(parameter, direction, side=side, chunk_size=2)
    assert returned is parameter and parameter.data_ptr() == pointer
    assert parameter.requires_grad and parameter.stride() == strides
    assert parameter.dtype == dtype and parameter.device == base.device
    assert torch.equal(alias, expected)


def test_rank_deficient_scaled_and_empty_directions():
    direction = torch.tensor([1.0, 2.0, -3.0, 4.0], dtype=torch.float64)
    dependent = torch.stack((direction, -2 * direction, torch.zeros_like(direction)), dim=1)
    weight = torch.eye(4, dtype=torch.float64)
    expected = project_weights(weight, direction)
    actual = project_weights(weight, dependent)
    torch.testing.assert_close(actual, expected)
    huge = project_weights(weight, dependent * 1e200)
    torch.testing.assert_close(huge, expected)
    for directions in (torch.empty((4, 0), dtype=torch.float64), torch.zeros(4, dtype=torch.float64)):
        copied = project_weights(weight, directions)
        assert torch.equal(copied, weight) and copied.data_ptr() != weight.data_ptr()
        assert project_weights_(weight, directions) is weight
    basis = _basis(dependent, 4, torch.float64, None)
    assert basis.shape == (4, 1)
    assert basis[basis[:, 0].abs().argmax(), 0] > 0


def test_neutral_covariance_matches_precomputed_without_mutation():
    generator = torch.Generator().manual_seed(31)
    neutral = torch.randn((19, 5), dtype=torch.float64, generator=generator)
    direction = torch.randn(5, dtype=torch.float64, generator=generator)
    weight = torch.eye(5, dtype=torch.float64)
    sigma = neutral.T @ neutral + 0.7 * weight
    original = sigma.clone()
    from_samples = project_weights(weight, direction, mode="covariance", neutral=neutral, regularization=0.7, chunk_size=3)
    from_sigma = project_weights(weight, direction, mode="covariance", covariance=sigma)
    torch.testing.assert_close(from_samples, from_sigma, rtol=1e-12, atol=1e-12)
    assert torch.equal(sigma, original)


def test_singular_psd_fallback_and_unsupported_subspace():
    weight = torch.eye(3, dtype=torch.float64)
    sigma = torch.diag(torch.tensor([2.0, 0.0, 3.0], dtype=torch.float64))
    direction = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    projector = project_weights(weight, direction, mode="covariance", covariance=sigma)
    basis = direction / direction.norm()
    solved = torch.linalg.pinv(sigma, hermitian=True) @ basis
    expected = weight - torch.outer(solved / (basis @ solved), basis)
    torch.testing.assert_close(projector, expected)
    original = weight.clone()
    with pytest.raises(ValueError, match="cannot support"):
        project_weights_(weight, torch.tensor([0.0, 1.0, 0.0]), mode="covariance", covariance=sigma)
    assert torch.equal(weight, original)


@pytest.mark.parametrize("scale", [torch.nextafter(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64)).item(), 1e-300, 1e300])
def test_extreme_covariance_scale_cancels(scale):
    weight = torch.eye(3, dtype=torch.float64)
    direction = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expected = project_weights(weight, direction)
    actual = project_weights(weight, direction, mode="covariance", covariance=weight * scale)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64])
def test_outer_autocast_does_not_change_projection(compute_dtype):
    generator = torch.Generator().manual_seed(37)
    weight = torch.randn((128, 7, 5), generator=generator)
    directions = torch.randn((5, 2), generator=generator)
    neutral = torch.randn((17, 5), generator=generator)
    options = dict(mode="covariance", neutral=neutral, compute_dtype=compute_dtype, chunk_size=3)
    baseline = project_weights(weight, directions, **options)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = project_weights(weight, directions, **options)
    assert torch.equal(actual, baseline)


@pytest.mark.parametrize("options", [
    {"mode": "bad"}, {"side": "bad"}, {"compute_dtype": torch.bfloat16},
    {"regularization": -1}, {"regularization": float("nan")},
    {"rcond": -1}, {"rcond": 1}, {"chunk_size": 0}, {"chunk_size": True},
    {"mode": "covariance"}, {"neutral": torch.ones(2, 3)},
    {"mode": "covariance", "neutral": torch.ones(2, 3), "covariance": torch.eye(3)},
    {"mode": "covariance", "covariance": torch.eye(2)},
    {"mode": "covariance", "covariance": torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])},
    {"mode": "covariance", "covariance": torch.diag(torch.tensor([1.0, -1.0, 1.0]))},
])
def test_invalid_options_fail_before_mutation(options):
    weight = torch.eye(3)
    original = weight.clone()
    with pytest.raises((ValueError, TypeError)):
        project_weights_(weight, torch.ones(3), **options)
    assert torch.equal(weight, original)


@pytest.mark.parametrize("bad_weight", [torch.ones(3), torch.ones(2, 3, 4, 5), torch.ones(2, 3, dtype=torch.int64), torch.empty(0, 3)])
def test_invalid_weights(bad_weight):
    with pytest.raises((ValueError, TypeError)):
        project_weights(bad_weight, torch.ones(3))


def test_nonfinite_late_source_chunk_fails_before_mutation():
    weight = torch.eye(7)
    weight[-1, -1] = float("nan")
    original = weight.clone()
    with pytest.raises(ValueError, match="NaN or infinity"):
        project_weights_(weight, torch.ones(7), chunk_size=2)
    torch.testing.assert_close(weight, original, equal_nan=True)


def test_overlapping_storage_rejected_but_copy_supported():
    weight = torch.ones(1, 5).expand(3, 5)
    with pytest.raises(ValueError, match="nonoverlapping"):
        project_weights_(weight, torch.ones(5))
    projected = project_weights(weight, torch.ones(5))
    torch.testing.assert_close(projected, torch.zeros_like(projected), atol=1e-6, rtol=0)


def test_direction_shape_and_nonfinite_validation():
    weight = torch.eye(3)
    for directions in (torch.ones(2), torch.ones(3, 1, 1), torch.tensor([1.0, float("inf"), 0.0])):
        with pytest.raises(ValueError):
            project_weights(weight, directions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batched_native_dtype():
    generator = torch.Generator(device="cuda").manual_seed(41)
    weight = torch.randn((128, 9, 5), device="cuda", dtype=torch.bfloat16, generator=generator)
    direction = torch.zeros(5, device="cuda")
    direction[0] = 1
    expected = weight.clone()
    expected[..., 0] = 0
    pointer = weight.data_ptr()
    project_weights_(weight, direction, compute_dtype=torch.float32, chunk_size=3)
    assert weight.data_ptr() == pointer and torch.equal(weight, expected)
