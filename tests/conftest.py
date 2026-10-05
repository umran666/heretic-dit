"""Shared fakes and fixtures for Heretic-DiT eval tests.

Everything runs on CPU with no downloads: tiny stand-in adapters, fake
classifiers, and in-memory backends replace all heavy dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heretic_dit.eval import CallableAdapter, DeterministicGenerativeValidator, ValidatorConfig  # noqa: E402
from heretic_dit.interfaces import RecoveryResult  # noqa: E402


class CountingAdapter:
    """Fake GenerationAdapter that counts generate() calls and returns
    deterministic images driven purely by (prompt, seed)."""

    def __init__(self, model_id: str = "fake/model", calls: dict | None = None) -> None:
        self.model_id = model_id
        self.calls = calls if calls is not None else {}

    def generate(self, prompt: str, seed: int, num_inference_steps: int, guidance_scale: float) -> torch.Tensor:
        self.calls[(prompt, seed, num_inference_steps)] = self.calls.get((prompt, seed, num_inference_steps), 0) + 1
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(3, 16, 16, generator=generator)


class KeywordClassifier:
    """Fake ConceptClassifier: high confidence when the *concept* keyword is in
    the fake mapping, else low. Deterministic and download-free."""

    def __init__(self, hit_value: float = 0.9, miss_value: float = 0.1, threshold_key: str = "") -> None:
        self.hit_value = hit_value
        self.miss_value = miss_value
        self._threshold_key = threshold_key

    def classify(self, images, concept: str):
        # images carries no text; the fake scores by concept membership only.
        count = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        value = self.hit_value if concept in CONCEPT_KEYWORDS else self.miss_value
        return [value] * count


CONCEPT_KEYWORDS = {"church", "tench", "van gogh", "tom hanks"}


class ConstantClassifier:
    """Scores every image with a fixed value per concept."""

    def __init__(self, value_by_concept: dict) -> None:
        self.value_by_concept = value_by_concept

    def classify(self, images, concept: str):
        count = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        return [self.value_by_concept.get(concept, 0.0)] * count


def make_validator(tmp_path=None, hit_value: float = 0.9) -> DeterministicGenerativeValidator:
    config = ValidatorConfig(
        num_inference_steps=4,
        neutral_samples=4,
        batch_size=2,
        cache_dir=tmp_path,
    )
    classifier = KeywordClassifier(hit_value=hit_value)
    concept_prompts = {
        "church": [f"a photo of a church {i}" for i in range(4)],
        "tench": [f"a photo of a tench {i}" for i in range(4)],
    }
    neutral = ["a scenic landscape", "a quiet beach", "a desk with books", "a mountain ridge"]
    return DeterministicGenerativeValidator(classifier, concept_prompts, neutral, config)


def make_budget(validator, tmp_path=None, **overrides):
    budget = {
        "validator": validator,
        "adapter_factory": lambda model: CallableAdapter(
            generate_fn=lambda prompt, seed, steps, guidance: torch.rand(3, 16, 16),  # type: ignore[arg-type,return-value]
            model_id=f"fake/{model}",
        ),
        "seed": 0,
        "num_samples": 4,
    }
    budget.update(overrides)
    return budget


def fake_recovery_result(method: str = "fake", recovery: float = 0.5) -> RecoveryResult:
    return RecoveryResult(
        concept="church",
        method=method,
        recovery_score=recovery,
        drift_score=0.0,
        metrics={},
        cost={},
    )


@pytest.fixture
def validator(tmp_path):
    return make_validator(tmp_path)
