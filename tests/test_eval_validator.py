"""Tests for the generative validator: protocol conformance, resumability,
paired-seed correctness, and the erased-vs-unerased gap."""

from __future__ import annotations

import torch

from heretic_dit.eval import (
    CallableAdapter,
    ValidatorConfig,
    generate_paired_images,
    recovery_gap,
    deterministic_image_seeds,
)
from heretic_dit.interfaces import ConceptClassifier, GenerativeValidator, RecoveryResult

from tests.conftest import CountingAdapter, KeywordClassifier, make_validator


def test_validator_satisfies_protocol(validator):
    assert isinstance(validator, GenerativeValidator)
    assert isinstance(KeywordClassifier(), ConceptClassifier)


def test_recovery_rate_and_drift_use_heldout_and_neutral_prompts():
    calls: dict = {}
    validator = make_validator()
    validator._classifier = ConstantLikeClassifier()
    adapter = CountingAdapter("fake/base", calls)
    result = validator.validate(adapter, "church", num_samples=6, seed=11)
    assert isinstance(result, RecoveryResult)
    # 6 concept images + 4 neutral images = 10 generate calls
    assert sum(calls.values()) == 10
    assert result.cost["sample_count"] == 10
    # classifier hits for the concept -> recovery 1.0; also hits on neutral -> drift 1.0
    assert result.recovery_score == 1.0
    assert result.drift_score == 1.0


class ConstantLikeClassifier(KeywordClassifier):
    pass


def test_missing_concept_raises():
    validator = make_validator()
    try:
        validator.validate(CountingAdapter("fake/base"), "nonexistent", 2, 0)
    except KeyError as error:
        assert "nonexistent" in str(error)
    else:
        raise AssertionError("expected KeyError")


def test_resumability_reuses_cached_images(tmp_path):
    calls: dict = {}
    validator = make_validator(tmp_path)
    adapter = CountingAdapter("fake/base", calls)
    first = validator.validate(adapter, "church", 4, seed=5)
    calls_after_first = dict(calls)
    assert calls_after_first  # generated something
    second = validator.validate(adapter, "church", 4, seed=5)
    assert calls == calls_after_first  # zero new generate calls
    assert second.recovery_score == first.recovery_score
    assert second.metrics["mean_confidence"] == first.metrics["mean_confidence"]


def test_cache_partial_resume(tmp_path):
    calls: dict = {}
    validator = make_validator(tmp_path)
    adapter = CountingAdapter("fake/base", calls)
    validator.validate(adapter, "church", 4, seed=5)
    # A different model id must NOT reuse the same cache entries.
    other = CountingAdapter("fake/edited", calls)
    validator.validate(other, "church", 4, seed=5)
    for (prompt, seed, steps), count in calls.items():
        assert count <= 1


def test_determinism_across_validator_instances(tmp_path):
    a = make_validator(tmp_path)
    b = make_validator(tmp_path)
    result_a = a.validate(CountingAdapter("fake/base"), "church", 4, seed=9)
    result_b = b.validate(CountingAdapter("fake/base"), "church", 4, seed=9)
    assert result_a.recovery_score == result_b.recovery_score
    assert result_a.metrics["mean_confidence"] == result_b.metrics["mean_confidence"]


def test_per_image_seeds_stable_and_model_separated():
    prompts = ["p1", "p2"]
    base = deterministic_image_seeds("fake/base", prompts, seed=3, num_inference_steps=4, guidance_scale=2.0)
    base_again = deterministic_image_seeds("fake/base", prompts, seed=3, num_inference_steps=4, guidance_scale=2.0)
    edited = deterministic_image_seeds("fake/edited", prompts, seed=3, num_inference_steps=4, guidance_scale=2.0)
    assert base == base_again
    assert base != edited
    # Adding a prompt must not shift the seeds of existing prompts.
    extended = deterministic_image_seeds("fake/base", prompts + ["p3"], seed=3, num_inference_steps=4, guidance_scale=2.0)
    assert extended[:2] == base


def test_paired_generation_uses_identical_run_seed(tmp_path):
    calls_a: dict = {}
    calls_b: dict = {}
    adapters = {
        "base": CountingAdapter("fake/base", calls_a),
        "edited": CountingAdapter("fake/edited", calls_b),
    }
    prompts = ["a landscape", "a beach"]
    images = generate_paired_images(adapters, prompts, seed=42, config=ValidatorConfig())
    assert set(images) == {"base", "edited"}
    assert images["base"].shape == images["edited"].shape == (2, 3, 16, 16)
    # Both adapters saw the same prompts with the same per-index run seed.
    assert {p for p, *_ in calls_a} == set(prompts)
    assert {p for p, *_ in calls_b} == set(prompts)
    seeds_a = {seed for _, seed, _ in calls_a}
    seeds_b = {seed for _, seed, _ in calls_b}
    assert seeds_a != seeds_b  # different model ids -> different derived seeds
    assert len(seeds_a) == len(prompts)


def test_recovery_gap_helper():
    base = fake_result("fake/base", recovery=0.9)
    edited = fake_result("fake/edited", recovery=0.4)
    assert recovery_gap(base, edited) == 0.5


def fake_result(model_id: str, recovery: float) -> RecoveryResult:
    return RecoveryResult(
        concept="church",
        method=model_id,
        recovery_score=recovery,
        drift_score=0.0,
        metrics={},
        cost={},
    )


def test_bad_images_rejected():
    def bad_fn(prompt, seed, steps, guidance):
        return torch.rand(3, 16, 16) * 4.0  # out of [0, 1]

    adapter = CallableAdapter(generate_fn=bad_fn, model_id="fake/bad")
    validator = make_validator()
    try:
        validator.validate(adapter, "church", 1, 0)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("expected ValueError for out-of-range images")
