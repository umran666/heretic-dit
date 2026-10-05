"""Tests for baseline interface conformance, cost accounting, and null baselines."""

from __future__ import annotations

import pytest
import torch

from heretic_dit.baselines import (
    FinetuneRecovery,
    NoEditBaseline,
    RandomProjectionBaseline,
    TextualInversionRecovery,
    UnerasedReferenceBaseline,
    make_null_baseline,
)
from heretic_dit.interfaces import BaselineRunner, RecoveryResult

from tests.conftest import make_validator

REQUIRED_COST_KEYS = ("wall_clock_sec", "peak_vram_mb", "trainable_params", "sample_count")


@pytest.fixture
def budget(tmp_path):
    return {
        "validator": make_validator(tmp_path),
        "adapter_factory": _fake_adapter_factory,
        "seed": 7,
        "num_samples": 4,
        "model_dim": 32,
        "apply_edit": _fake_apply_edit,
        "alpha": 0.75,
    }


def _fake_adapter_factory(model):
    from heretic_dit.eval import CallableAdapter

    def generate(prompt, seed, steps, guidance):
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(3, 16, 16, generator=generator)

    return CallableAdapter(generate_fn=generate, model_id=f"fake/{model}")


def _fake_apply_edit(model, edit_spec):
    assert edit_spec.subspace is not None
    assert edit_spec.subspace.shape[1] > 0
    # Orthonormality of the matched random subspace.
    gram = edit_spec.subspace.T @ edit_spec.subspace
    assert torch.allclose(gram, torch.eye(gram.shape[0]), atol=1e-5)
    return f"projected/{model}"


@pytest.mark.parametrize(
    "runner",
    [
        NoEditBaseline(),
        RandomProjectionBaseline(dim=4),
        UnerasedReferenceBaseline(),
        TextualInversionRecovery(),
        FinetuneRecovery(mode="lora"),
        FinetuneRecovery(mode="full"),
        make_null_baseline("no-edit"),
        make_null_baseline("random-projection"),
        make_null_baseline("unerased-reference"),
    ],
)
def test_all_baselines_conform_to_protocol(runner, budget):
    assert isinstance(runner, BaselineRunner)
    model = "ref" if runner.method_name == "unerased-reference" else "erased"
    result = runner.run(model, "church", budget)
    assert isinstance(result, RecoveryResult)
    assert result.concept == "church"
    assert result.method == runner.method_name
    assert 0.0 <= result.recovery_score <= 1.0
    assert 0.0 <= result.drift_score <= 1.0
    for key in REQUIRED_COST_KEYS:
        assert key in result.cost, (runner.method_name, key)


def test_null_baseline_factory_rejects_unknown_kind():
    with pytest.raises(KeyError, match="Unknown null baseline"):
        make_null_baseline("does-not-exist")


def test_random_projection_matches_alpha(tmp_path):
    captured = {}

    def apply_edit(model, edit_spec):
        captured["alpha"] = edit_spec.alpha
        captured["seed_meta"] = edit_spec.metadata["seed"]
        return f"projected/{model}"

    budget = {
        "validator": make_validator(tmp_path),
        "adapter_factory": _fake_adapter_factory,
        "seed": 11,
        "apply_edit": apply_edit,
        "alpha": 0.3,
        "model_dim": 16,
    }
    RandomProjectionBaseline(dim=2).run("erased", "church", budget)
    assert captured["alpha"] == 0.3
    assert captured["seed_meta"] == 11


def test_missing_budget_keys_raise_actionable_errors(tmp_path):
    runner = NoEditBaseline()
    with pytest.raises(KeyError, match="adapter_factory"):
        runner.run("m", "church", {"validator": make_validator(tmp_path)})


def test_training_baselines_report_trainable_params_and_steps(tmp_path):
    budget = {
        "validator": make_validator(tmp_path),
        "adapter_factory": _fake_adapter_factory,
        "seed": 2,
        "steps": 10,
    }
    ti = TextualInversionRecovery().run("erased", "church", budget)
    assert ti.cost["trainable_params"] > 0
    assert ti.cost["steps"] == 10
    assert ti.cost["training_images"] == 5  # default few-shot prompt count
    lora = FinetuneRecovery(mode="lora").run("erased", "church", budget)
    assert lora.cost["trainable_params"] > 0
    assert lora.cost["steps"] == 10


def test_invalid_budget_values_rejected(tmp_path):
    budget = {
        "validator": make_validator(tmp_path),
        "adapter_factory": _fake_adapter_factory,
        "steps": 0,
    }
    with pytest.raises(ValueError, match="steps"):
        TextualInversionRecovery().run("erased", "church", budget)
    with pytest.raises(ValueError, match="mode"):
        FinetuneRecovery(mode="sgd")
