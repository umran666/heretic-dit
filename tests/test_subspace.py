"""Deterministic concept-subspace extraction and covariance regression tests."""

import math

import pytest
import torch

from heretic_dit.core.projector import project_weights
from heretic_dit.core.subspace import (
    aggregate_activations,
    compute_covariance,
    contrastive_pca,
    mean_difference,
    subspace_svd,
)


def _diagonal_samples(variances):
    """Symmetric samples with exactly the supplied population covariance."""
    values = torch.tensor(variances, dtype=torch.float64)
    positive = torch.diag(values.sqrt()) * math.sqrt(values.numel())
    return torch.cat((positive, -positive))


def _assert_basis(basis):
    torch.testing.assert_close(
        basis.T @ basis,
        torch.eye(basis.shape[1], dtype=basis.dtype),
        atol=1e-12 if basis.dtype == torch.float64 else 1e-5,
        rtol=1e-12 if basis.dtype == torch.float64 else 1e-5,
    )
    if basis.shape[1]:
        pivots = basis.abs().argmax(dim=0)
        assert bool((basis[pivots, torch.arange(basis.shape[1])] > 0).all())


def test_mean_difference_preserves_target_minus_neutral_orientation():
    neutral = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], dtype=torch.float64)
    offset = torch.tensor([-3.0, 1.0, -2.0], dtype=torch.float64)
    target = neutral + offset
    actual = mean_difference(target, neutral)
    assert actual.shape == (3, 1)
    torch.testing.assert_close(actual[:, 0], offset / offset.norm())
    torch.testing.assert_close(actual.norm(), torch.tensor(1.0, dtype=torch.float64))
    torch.testing.assert_close(mean_difference(neutral, target), -actual)


def test_mean_difference_allows_different_population_sizes():
    target = torch.tensor([[4.0, 1.0], [2.0, -1.0]], dtype=torch.float64)
    neutral = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        mean_difference(target, neutral), torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    )


def test_mean_difference_rejects_numerically_empty_direction():
    samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    with pytest.raises(ValueError):
        mean_difference(samples, samples.clone())
    with pytest.raises(ValueError):
        mean_difference(samples + 1e-8, samples, rcond=1e-3)


@pytest.mark.parametrize("dtype, offset", [(torch.float32, 1e6), (torch.float64, 1e15)])
def test_mean_difference_preserves_orientation_with_large_shared_offset(dtype, offset):
    target = torch.tensor([[offset + 1.0, offset + 2.0]], dtype=dtype).repeat(3, 1)
    neutral = torch.tensor([[offset, offset]], dtype=dtype).repeat(5, 1)
    expected = torch.tensor([[1.0], [2.0]], dtype=dtype) / math.sqrt(5)
    direction = mean_difference(target, neutral, compute_dtype=dtype)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(direction, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype, offset", [(torch.float32, 1e30), (torch.float64, 1e300)])
def test_mean_difference_retains_direction_beside_huge_shared_constant_feature(dtype, offset):
    target = torch.tensor([[offset, 1.0], [offset, 1.0]], dtype=dtype)
    neutral = torch.tensor([[offset, 0.0], [offset, 0.0]], dtype=dtype)
    direction = mean_difference(target, neutral, compute_dtype=dtype)
    torch.testing.assert_close(direction, torch.tensor([[0.0], [1.0]], dtype=dtype),
                               atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(6, 4), (2, 3, 4), (1, 2, 3, 4)])
def test_unweighted_aggregation_flattens_every_sample_axis(shape):
    samples = torch.arange(math.prod(shape), dtype=torch.bfloat16).reshape(shape)
    original = samples.clone()
    actual = aggregate_activations(samples)
    assert actual.shape == (math.prod(shape[:-1]), shape[-1])
    assert actual.device.type == "cpu" and actual.dtype == torch.float64
    assert actual.data_ptr() != samples.data_ptr()
    torch.testing.assert_close(actual, samples.reshape(-1, shape[-1]).double())
    assert torch.equal(samples, original)


def test_timestep_weights_are_literal_sums_before_statistics():
    samples = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]],
          [[100.0, 200.0], [300.0, 400.0]]]], dtype=torch.float64
    )
    weights = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    expected = (0.5 * samples[:, 1] + 2 * samples[:, 2]).reshape(-1, 2)
    actual = aggregate_activations(samples, timestep_weights=weights)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        aggregate_activations(samples, timestep_weights=2 * weights), 2 * expected
    )
    torch.testing.assert_close(
        compute_covariance(samples, shrinkage=0.0, regularization=0.0,
                           timestep_weights=weights),
        expected.T @ expected / expected.shape[0],
    )


def test_semantic_timestep_window_excludes_unselected_steps():
    generator = torch.Generator().manual_seed(101)
    target = torch.randn((2, 10, 3, 4), generator=generator, dtype=torch.float64)
    neutral = target.clone()
    target[:, 4:9, :, 2] += 2
    target[:, :4, :, 0] += 100
    target[:, 9:, :, 0] += 100
    weights = torch.tensor([0.0] * 4 + [0.2] * 5 + [0.0], dtype=torch.float64)
    direction = mean_difference(target, neutral, timestep_weights=weights)
    expected = torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=torch.float64)
    torch.testing.assert_close(direction, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", [subspace_svd, compute_covariance])
def test_weighted_statistics_match_explicit_aggregated_samples(method):
    generator = torch.Generator().manual_seed(103)
    samples = torch.randn((2, 4, 3, 5), generator=generator, dtype=torch.float64)
    weights = torch.tensor([0.0, 1.0, 2.0, 0.0], dtype=torch.float64)
    explicit = (samples[:, 1] + 2 * samples[:, 2]).reshape(-1, 5)
    expected = method(explicit)
    actual = method(samples, timestep_weights=weights)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_contrastive_pca_weights_match_explicit_population_aggregation():
    generator = torch.Generator().manual_seed(107)
    target = torch.randn((2, 4, 3, 5), generator=generator, dtype=torch.float64)
    neutral = torch.randn((3, 4, 3, 5), generator=generator, dtype=torch.float64)
    weights = torch.tensor([0.0, 1.0, 0.0, 0.5], dtype=torch.float64)
    expected = contrastive_pca(
        (target[:, 1] + 0.5 * target[:, 3]).reshape(-1, 5),
        (neutral[:, 1] + 0.5 * neutral[:, 3]).reshape(-1, 5), k=2,
    )
    actual = contrastive_pca(target, neutral, k=2, timestep_weights=weights)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("threshold, rank", [(0.85, 1), (0.95, 2), (1.0, 2)])
def test_svd_uses_squared_singular_values_and_returns_numerical_rank(threshold, rank):
    samples = _diagonal_samples([9.0, 1.0, 0.0, 0.0]) + 37
    basis = subspace_svd(samples, energy_threshold=threshold, max_rank=4)
    assert basis.shape == (4, rank)
    _assert_basis(basis)
    torch.testing.assert_close(basis, torch.eye(4, dtype=torch.float64)[:, :rank])


def test_svd_rank_cap_and_captured_variance():
    generator = torch.Generator().manual_seed(109)
    orthogonal, _ = torch.linalg.qr(
        torch.randn((6, 6), generator=generator, dtype=torch.float64)
    )
    samples = _diagonal_samples([8.0, 4.0, 2.0, 1.0, 0.0, 0.0]) @ orthogonal.T
    basis = subspace_svd(samples, energy_threshold=0.99, max_rank=2)
    assert basis.shape == (6, 2)
    _assert_basis(basis)
    torch.testing.assert_close(basis @ basis.T, orthogonal[:, :2] @ orthogonal[:, :2].T)
    total_variance = samples.square().sum()
    captured_variance = (samples @ basis).square().sum()
    torch.testing.assert_close(captured_variance / total_variance, torch.tensor(12 / 15, dtype=torch.float64))


def test_svd_centering_single_sample_and_zero_variance():
    for samples in (torch.ones((7, 3)), torch.ones((1, 3)), torch.zeros((5, 3))):
        assert subspace_svd(samples).shape == (3, 0)
    uncentered = subspace_svd(torch.ones((7, 3)), center=False)
    assert uncentered.shape == (3, 1)
    _assert_basis(uncentered)
    torch.testing.assert_close(uncentered[:, 0], torch.full((3,), 1 / math.sqrt(3), dtype=torch.float64))


def test_svd_rcond_removes_tiny_variance():
    samples = _diagonal_samples([1.0, 1e-14, 0.0])
    basis = subspace_svd(samples, energy_threshold=1.0, max_rank=3, rcond=1e-6)
    assert basis.shape == (3, 1)


def test_contrastive_pca_selects_positive_eigenvalues_in_algebraic_order():
    target = _diagonal_samples([4.0, 3.0, 1.0]) + 15
    neutral = _diagonal_samples([0.0, 2.0, 3.0]) - 21
    basis = contrastive_pca(target, neutral, k=3)
    assert basis.shape == (3, 2)
    _assert_basis(basis)
    torch.testing.assert_close(basis, torch.eye(3, dtype=torch.float64)[:, :2])
    top = contrastive_pca(target, neutral, k=1)
    torch.testing.assert_close(top, torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64))


def test_contrastive_pca_alpha_controls_neutral_variance_penalty():
    target = _diagonal_samples([3.0, 2.0])
    neutral = _diagonal_samples([3.0, 0.0])
    torch.testing.assert_close(
        contrastive_pca(target, neutral, alpha=0),
        torch.tensor([[1.0], [0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        contrastive_pca(target, neutral, alpha=1),
        torch.tensor([[0.0], [1.0]], dtype=torch.float64),
    )


def test_contrastive_pca_empty_when_no_positive_contrast():
    samples = _diagonal_samples([1.0, 2.0, 3.0])
    assert contrastive_pca(samples, samples, k=3).shape == (3, 0)
    assert contrastive_pca(samples, 2 * samples, k=3).shape == (3, 0)
    assert contrastive_pca(torch.ones((1, 3)), torch.ones((2, 3))).shape == (3, 0)


def test_contrastive_pca_uncentered_second_moment_can_capture_mean():
    target = torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    neutral = torch.zeros((3, 2), dtype=torch.float64)
    assert contrastive_pca(target, neutral).shape == (2, 0)
    torch.testing.assert_close(
        contrastive_pca(target, neutral, center=False),
        torch.tensor([[1.0], [0.0]], dtype=torch.float64),
    )


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("shrinkage", [0.0, 0.25, 1.0, "diagonal"])
def test_covariance_population_moment_shrinkage_and_additive_ridge(center, shrinkage):
    samples = torch.tensor([[1.0, 3.0, 2.0], [2.0, -1.0, 0.0], [4.0, 2.0, 1.0]], dtype=torch.float64)
    data = samples - samples.mean(dim=0) if center else samples
    empirical = data.T @ data / data.shape[0]
    if shrinkage == "diagonal":
        expected = torch.diag(empirical.diagonal())
    else:
        isotropic = empirical.trace() / empirical.shape[0] * torch.eye(3, dtype=torch.float64)
        expected = (1 - shrinkage) * empirical + shrinkage * isotropic
    ridge = 0.125
    expected += ridge * torch.eye(3, dtype=torch.float64)
    actual = compute_covariance(samples, shrinkage=shrinkage, center=center, regularization=ridge)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual, actual.T, atol=0, rtol=0)
    assert float(torch.linalg.eigvalsh(actual).min()) >= ridge - 1e-12


@pytest.mark.parametrize("center", [False, True])
def test_ledoit_wolf_matches_independent_outer_product_variance_reference(center):
    samples = torch.tensor([[1.0, -1.0, 2.0], [3.0, 0.0, 1.0], [0.0, 2.0, -1.0],
                            [-2.0, 1.0, 0.5], [1.0, 3.0, 2.0]], dtype=torch.float64)
    data = samples - samples.mean(dim=0) if center else samples
    n, d = data.shape
    empirical = data.T @ data / n
    isotropic = empirical.trace() / d * torch.eye(d, dtype=torch.float64)
    target_distance = (empirical - isotropic).square().sum()
    # This direct statistical definition is independent of optimized LW formulas.
    variance = sum((torch.outer(sample, sample) - empirical).square().sum() for sample in data) / n**2
    coefficient = min(float(variance / target_distance), 1.0)
    expected = (1 - coefficient) * empirical + coefficient * isotropic + 0.2 * torch.eye(d, dtype=torch.float64)
    actual = compute_covariance(samples, shrinkage="ledoit_wolf", center=center, regularization=0.2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shrinkage", ["ledoit_wolf", "diagonal", 0.0, 1.0])
def test_zero_covariance_regularizes_without_nan(shrinkage):
    actual = compute_covariance(torch.zeros((4, 3)), shrinkage=shrinkage, regularization=0.3)
    torch.testing.assert_close(actual, 0.3 * torch.eye(3, dtype=torch.float64))


def test_covariance_single_feature_and_single_sample():
    torch.testing.assert_close(
        compute_covariance(torch.tensor([[1.0], [3.0]]), regularization=0),
        torch.tensor([[5.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        compute_covariance(torch.tensor([[2.0, 3.0]]), regularization=0.1, center=True),
        0.1 * torch.eye(2, dtype=torch.float64),
    )


@pytest.mark.parametrize("dtype, offset", [(torch.float32, 1e30), (torch.float64, 1e300)])
def test_centered_statistics_preserve_small_variance_beside_huge_constant_offset(dtype, offset):
    target = torch.tensor([[offset, -1.0], [offset, 1.0]], dtype=dtype)
    neutral = torch.tensor([[offset, 0.0], [offset, 0.0]], dtype=dtype)
    offset_free = torch.tensor([[0.0, -1.0], [0.0, 1.0]], dtype=dtype)
    expected_basis = torch.tensor([[0.0], [1.0]], dtype=dtype)
    expected_covariance = torch.diag(torch.tensor([0.0, 1.0], dtype=dtype))
    empirical = compute_covariance(target, shrinkage=0.0, regularization=0.0,
                                   center=True, compute_dtype=dtype)
    torch.testing.assert_close(empirical, expected_covariance, atol=0, rtol=0)
    shrunk = compute_covariance(target, shrinkage="ledoit_wolf", regularization=0.125,
                                center=True, compute_dtype=dtype)
    reference = compute_covariance(offset_free, shrinkage="ledoit_wolf", regularization=0.125,
                                   center=True, compute_dtype=dtype)
    torch.testing.assert_close(shrunk, reference, atol=0, rtol=0)
    torch.testing.assert_close(
        contrastive_pca(target, neutral, compute_dtype=dtype), expected_basis, atol=0, rtol=0
    )
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(subspace_svd(target, compute_dtype=dtype), expected_basis,
                               atol=tolerance, rtol=tolerance)


def test_contrastive_alpha_must_be_representable_in_compute_dtype():
    samples = _diagonal_samples([1.0, 2.0])
    with pytest.raises(ValueError):
        contrastive_pca(samples, samples, alpha=1e308, compute_dtype=torch.float32)


@pytest.mark.parametrize("dtype, offset", [(torch.float32, 1e6), (torch.float64, 1e15)])
def test_centered_covariance_preserves_exact_moderate_offset_residuals(dtype, offset):
    samples = torch.tensor([[offset - 1.0], [offset + 1.0]], dtype=dtype)
    actual = compute_covariance(samples, shrinkage=0.0, regularization=0.0,
                                center=True, compute_dtype=dtype)
    torch.testing.assert_close(actual, torch.ones((1, 1), dtype=dtype), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_contrastive_pca_repeated_empirical_distribution_has_no_positive_subspace(dtype):
    generator = torch.Generator().manual_seed(137)
    samples = torch.randn((7, 4), generator=generator, dtype=dtype)
    basis = contrastive_pca(samples, samples.repeat(3, 1), k=4, compute_dtype=dtype)
    assert basis.shape == (4, 0)


def test_extracted_basis_and_covariance_integrate_with_projector():
    generator = torch.Generator().manual_seed(113)
    neutral = torch.randn((31, 5), generator=generator, dtype=torch.float64)
    target = neutral + torch.tensor([1.0, -2.0, 0.0, 0.0, 1.0], dtype=torch.float64)
    basis = mean_difference(target, neutral)
    sigma = compute_covariance(neutral, shrinkage=0.25, regularization=0.5)
    identity = torch.eye(5, dtype=torch.float64)
    projector = project_weights(identity, basis, mode="covariance", covariance=sigma)
    solved = torch.linalg.solve(sigma, basis)
    expected = identity - solved @ torch.linalg.solve(basis.T @ solved, basis.T)
    torch.testing.assert_close(projector, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(basis.T @ projector, torch.zeros((1, 5), dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_outputs_are_detached_cpu_and_repeatable_under_outer_autocast(dtype):
    generator = torch.Generator().manual_seed(127)
    samples = torch.randn((19, 5), generator=generator, dtype=torch.bfloat16).requires_grad_()
    neutral = torch.randn((23, 5), generator=generator, dtype=torch.bfloat16).requires_grad_()
    originals = samples.detach().clone(), neutral.detach().clone()
    functions = (
        lambda: aggregate_activations(samples, compute_dtype=dtype),
        lambda: mean_difference(samples, neutral, compute_dtype=dtype),
        lambda: subspace_svd(samples, compute_dtype=dtype),
        lambda: contrastive_pca(samples, neutral, k=2, compute_dtype=dtype),
        lambda: compute_covariance(samples, compute_dtype=dtype),
    )
    for operation in functions:
        expected = operation()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = operation()
        assert actual.device.type == "cpu" and actual.dtype == dtype
        assert not actual.requires_grad and actual.grad_fn is None
        assert torch.equal(actual, expected)
        assert torch.equal(operation(), expected)
    assert torch.equal(samples, originals[0]) and torch.equal(neutral, originals[1])


@pytest.mark.parametrize("bad", [torch.ones(3), torch.ones(1, 2, 3, 4, 5),
                                  torch.empty((0, 3)), torch.empty((2, 0)),
                                  torch.ones((2, 3), dtype=torch.int64),
                                  torch.ones((2, 3), dtype=torch.complex64),
                                  torch.tensor([[1.0, float("nan")]]),
                                  torch.tensor([[float("inf"), 0.0]])])
def test_invalid_activations_rejected_by_all_extractors(bad):
    operations = (
        lambda: aggregate_activations(bad),
        lambda: mean_difference(bad, torch.ones((2, bad.shape[-1]))),
        lambda: subspace_svd(bad),
        lambda: contrastive_pca(bad, torch.ones((2, bad.shape[-1]))),
        lambda: compute_covariance(bad),
    )
    for operation in operations:
        with pytest.raises((TypeError, ValueError)):
            operation()


@pytest.mark.parametrize("weights", [torch.ones(2), torch.ones((3, 1)),
                                      torch.tensor([1.0, -1.0, 1.0]), torch.zeros(3),
                                      torch.tensor([1.0, float("nan"), 0.0]),
                                      torch.tensor([1.0, float("inf"), 0.0])])
def test_invalid_timestep_weights(weights):
    with pytest.raises((TypeError, ValueError)):
        aggregate_activations(torch.ones((2, 3, 4, 5)), timestep_weights=weights)


@pytest.mark.parametrize("shape", [(6, 5), (2, 3, 5)])
def test_timestep_weights_require_an_explicit_timestep_axis(shape):
    with pytest.raises(ValueError):
        aggregate_activations(torch.ones(shape), timestep_weights=torch.ones(3))


@pytest.mark.parametrize("operation", [
    lambda x: aggregate_activations(x, compute_dtype=torch.bfloat16),
    lambda x: mean_difference(x, x + 1, rcond=-1),
    lambda x: mean_difference(x, x + 1, rcond=1),
    lambda x: subspace_svd(x, energy_threshold=0),
    lambda x: subspace_svd(x, energy_threshold=1.1),
    lambda x: subspace_svd(x, energy_threshold=float("nan")),
    lambda x: subspace_svd(x, max_rank=0),
    lambda x: subspace_svd(x, max_rank=True),
    lambda x: subspace_svd(x, rcond=float("nan")),
    lambda x: contrastive_pca(x, x, k=0),
    lambda x: contrastive_pca(x, x, k=4),
    lambda x: contrastive_pca(x, x, k=True),
    lambda x: contrastive_pca(x, x, alpha=-1),
    lambda x: contrastive_pca(x, x, alpha=float("inf")),
    lambda x: compute_covariance(x, shrinkage="unknown"),
    lambda x: compute_covariance(x, shrinkage=-0.1),
    lambda x: compute_covariance(x, shrinkage=1.1),
    lambda x: compute_covariance(x, shrinkage=float("nan")),
    lambda x: compute_covariance(x, regularization=-1),
    lambda x: compute_covariance(x, regularization=float("inf")),
])
def test_invalid_scalar_options(operation):
    with pytest.raises((TypeError, ValueError)):
        operation(torch.ones((5, 3)))


@pytest.mark.parametrize("method", [mean_difference, contrastive_pca])
def test_population_feature_dimensions_must_agree(method):
    with pytest.raises(ValueError):
        method(torch.ones((5, 3)), torch.ones((6, 4)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_activations_are_extracted_on_cpu_without_mutation():
    generator = torch.Generator(device="cuda").manual_seed(131)
    samples = torch.randn((17, 4), dtype=torch.bfloat16, device="cuda", generator=generator)
    original = samples.clone()
    cpu = samples.cpu()
    for operation in (subspace_svd, compute_covariance):
        actual = operation(samples)
        assert actual.device.type == "cpu" and actual.dtype == torch.float64
        assert torch.equal(actual, operation(cpu))
    assert torch.equal(samples, original)
