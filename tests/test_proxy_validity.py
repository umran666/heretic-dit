"""Tests for quality metrics (FID math on synthetic features) and proxy validity."""

from __future__ import annotations

import pytest
import torch

from heretic_dit.eval import (
    ProxyTrial,
    RandomProjectionFeatures,
    evaluate_proxy_validity,
    fid_score,
    frechet_distance,
    load_trials,
    save_trials,
    spearman,
)
from heretic_dit.eval.quality import _moment_statistics


def test_frechet_distance_identical_gaussians_is_zero():
    features = torch.randn(64, 8, dtype=torch.float64)
    mu, sigma = _moment_statistics(features)
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-8)


def test_frechet_distance_is_symmetric_and_grows_with_separation():
    a = torch.randn(64, 8, dtype=torch.float64)
    b = torch.randn(64, 8, dtype=torch.float64) + 10.0
    mu_a, sigma_a = _moment_statistics(a)
    mu_b, sigma_b = _moment_statistics(b)
    ab = frechet_distance(mu_a, sigma_a, mu_b, sigma_b)
    ba = frechet_distance(mu_b, sigma_b, mu_a, sigma_a)
    assert ab == pytest.approx(ba, rel=1e-8)
    assert ab > 10.0  # means are 10 apart per-dimension


def test_fid_identical_image_sets_is_zero():
    images = torch.rand(8, 3, 8, 8)
    extractor = RandomProjectionFeatures(out_dim=16)
    assert fid_score(images, images.clone(), extractor) == pytest.approx(0.0, abs=1e-6)


def test_fid_different_sets_is_positive():
    extractor = RandomProjectionFeatures(out_dim=16)
    value = fid_score(torch.rand(8, 3, 8, 8), torch.rand(8, 3, 8, 8), extractor)
    assert value > 0.0
    assert value == value  # not NaN


def test_random_projection_features_are_deterministic():
    images = torch.rand(4, 3, 8, 8)
    a = RandomProjectionFeatures(out_dim=8, seed=1).extract(images)
    b = RandomProjectionFeatures(out_dim=8, seed=1).extract(images)
    assert torch.equal(a, b)
    assert a.shape == (4, 8)


# ---------------------------------------------------------------------------
# Spearman helper on synthetic data
# ---------------------------------------------------------------------------


def test_spearman_perfect_monotonic_is_one():
    rho, pvalue = spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert rho == pytest.approx(1.0)
    assert pvalue <= 0.05


def test_spearman_inverted_is_minus_one():
    rho, _ = spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
    assert rho == pytest.approx(-1.0)


def test_spearman_handles_ties():
    rho, _ = spearman([1, 2, 2, 4, 5], [10, 20, 25, 40, 50])
    assert 0.95 < rho <= 1.0


def test_spearman_input_validation():
    with pytest.raises(ValueError, match="mismatch"):
        spearman([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="at least 3"):
        spearman([1, 2], [1, 2])
    with pytest.raises(ValueError, match="constant"):
        spearman([1, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        spearman([1, 2, float("nan")], [1, 2, 3])


def test_proxy_validity_on_perfect_proxy_is_trustworthy():
    trials = [
        ProxyTrial(
            proxy_recovery=r / 10,
            proxy_drift=1 - r / 10,
            real_recovery=r / 10,
            real_quality_loss=1 - r / 10,
            trial_id=i,
        )
        for i, r in enumerate(range(1, 11))
    ]
    report = evaluate_proxy_validity(trials)
    assert report.n_trials == 10
    assert report.recovery_rho == pytest.approx(1.0)
    assert report.quality_rho == pytest.approx(1.0)
    assert report.utility_rho == pytest.approx(1.0)
    assert report.trustworthy


def test_proxy_validity_on_inverted_proxy_is_rejected():
    trials = [
        ProxyTrial(
            proxy_recovery=r / 10,
            proxy_drift=1 - r / 10,
            real_recovery=1 - r / 10,  # proxy says good, reality says bad
            real_quality_loss=r / 10,
        )
        for r in range(1, 11)
    ]
    report = evaluate_proxy_validity(trials)
    assert report.recovery_rho == pytest.approx(-1.0)
    assert not report.trustworthy


def test_proxy_validity_on_noisy_proxy_is_partial():
    # Proxy ranks are a noisy, partially-correlated version of reality.
    proxy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    real = [0.5, 0.1, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4]
    trials = [
        ProxyTrial(proxy_recovery=p, proxy_drift=1 - p, real_recovery=r, real_quality_loss=1 - r)
        for p, r in zip(proxy, real)
    ]
    report = evaluate_proxy_validity(trials)
    assert 0.2 < report.recovery_rho < 0.7
    assert not report.trustworthy


def test_proxy_validity_utility_weights():
    # Proxy drift is anti-correlated with real quality loss, so weighting
    # drift into the utility must change the utility ranking.
    trials = [
        ProxyTrial(0.2, 0.9, 0.2, 0.1),
        ProxyTrial(0.4, 0.1, 0.4, 0.9),
        ProxyTrial(0.6, 0.5, 0.6, 0.5),
        ProxyTrial(0.8, 0.7, 0.8, 0.7),
    ]
    balanced = evaluate_proxy_validity(trials, utility_recovery_weight=1.0, utility_drift_weight=1.0)
    recovery_only = evaluate_proxy_validity(trials, utility_recovery_weight=1.0, utility_drift_weight=1e-9)
    # Ignoring drift makes the utility perfectly correlated (both sides
    # collapse onto recovery); the balanced utility cannot be.
    assert recovery_only.utility_rho == pytest.approx(1.0)
    assert balanced.utility_rho != recovery_only.utility_rho
    with pytest.raises(ValueError):
        evaluate_proxy_validity(trials, utility_recovery_weight=0.0)


def test_trial_json_roundtrip(tmp_path):
    trials = [ProxyTrial(0.5, 0.5, 0.9, 0.1, trial_id=3, label="esd/church", extra={"alpha": 0.5})]
    path = save_trials(trials, tmp_path / "trials.json")
    loaded = load_trials(path)
    assert len(loaded) == 1
    assert loaded[0].trial_id == 3
    assert loaded[0].extra["alpha"] == 0.5
    assert loaded[0].proxy_recovery == 0.5
