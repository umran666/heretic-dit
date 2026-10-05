"""Generative ground-truth validation with fixed seeds, batching, resumability.

Implements ``heretic_dit.interfaces.GenerativeValidator``:

    validate(self, model, concept, num_samples, seed) -> RecoveryResult

``model`` is anything satisfying the local :class:`GenerationAdapter`
protocol (see ``heretic_dit.eval.adapters``); a bare adapter also works.

Recovery is measured on each concept's *held-out* prompt split (prompts never
seen by the Optuna proxy search), so proxy overfitting shows up as a gap
between proxy and held-out recovery. Drift is the false-positive rate of the
classifier on the neutral control prompts, generated with the same
deterministic per-image seeding scheme.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch import Tensor

from heretic_dit.eval.adapters import (
    CallableAdapter,
    GenerationAdapter,
    deterministic_image_seeds,
)
from heretic_dit.interfaces import ConceptClassifier, RecoveryResult

__all__ = [
    "ValidatorConfig",
    "DeterministicGenerativeValidator",
    "resolve_adapter",
    "generate_paired_images",
    "recovery_gap",
]


@dataclass(frozen=True)
class ValidatorConfig:
    """Settings for :class:`DeterministicGenerativeValidator`."""

    num_inference_steps: int = 8
    guidance_scale: float = 3.0
    recovery_threshold: float = 0.5
    batch_size: int = 8
    cache_dir: Optional[Path] = None
    neutral_samples: int = 16


def resolve_adapter(model: Any) -> GenerationAdapter:
    """Accept an adapter, a ``CallableAdapter``-style callable, or an adapter-like object."""
    if isinstance(model, CallableAdapter) or hasattr(model, "generate") and hasattr(model, "model_id"):
        return model
    raise TypeError(
        "validate() expects a GenerationAdapter (heretic_dit.eval.adapters); got "
        f"{type(model)!r}. Wrap a callable with CallableAdapter or use DiffusersAdapter."
    )


class DeterministicGenerativeValidator:
    """Fixed-seed, batched, resumable generative recovery validation."""

    def __init__(
        self,
        classifier: ConceptClassifier,
        concept_prompts: Mapping[str, Sequence[str]],
        neutral_prompts: Sequence[str],
        config: ValidatorConfig = ValidatorConfig(),
    ) -> None:
        """Args:
        classifier: any ``ConceptClassifier``; drives recovery decisions.
        concept_prompts: concept name -> *held-out* prompts used for recovery.
        neutral_prompts: control prompts (concept-free) for drift.
        config: sampler and caching settings.
        """
        self._classifier = classifier
        self._concept_prompts = {k: list(v) for k, v in concept_prompts.items()}
        if not self._concept_prompts:
            raise ValueError("concept_prompts must contain at least one concept.")
        for name, prompts in self._concept_prompts.items():
            if not prompts:
                raise ValueError(f"Concept {name!r} has no held-out prompts.")
        if not neutral_prompts:
            raise ValueError("neutral_prompts must not be empty.")
        self._neutral_prompts = list(neutral_prompts)
        self._config = config
        if config.cache_dir is not None:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Image generation (resumable)
    # ------------------------------------------------------------------

    def _cache_path(self, model_id: str, prompt: str, image_seed: int) -> Optional[Path]:
        if self._config.cache_dir is None:
            return None
        key = hashlib.sha256(
            f"{model_id}|{prompt}|{image_seed}|{self._config.num_inference_steps}|"
            f"{self._config.guidance_scale}".encode("utf-8")
        ).hexdigest()
        return Path(self._config.cache_dir) / f"{key}.pt"

    def generate_images(
        self,
        adapter: GenerationAdapter,
        prompts: Sequence[str],
        seed: int,
    ) -> Tensor:
        """Generate one image per prompt, reusing cached images when present.

        Per-image seeds are derived deterministically from the run seed, the
        model id, and the prompt (see ``deterministic_image_seeds``), so a
        resumed run reproduces exactly the images a fresh run would have made.
        """
        seeds = deterministic_image_seeds(
            adapter.model_id,
            list(prompts),
            seed,
            self._config.num_inference_steps,
            self._config.guidance_scale,
        )
        images: List[Tensor] = [None] * len(prompts)  # type: ignore[list-item]
        missing: List[int] = []
        for index, (prompt, image_seed) in enumerate(zip(prompts, seeds)):
            path = self._cache_path(adapter.model_id, prompt, image_seed)
            if path is not None and path.exists():
                images[index] = torch.load(path, weights_only=True)
            else:
                missing.append(index)
        for start in range(0, len(missing), self._config.batch_size):
            for index in missing[start : start + self._config.batch_size]:
                image = adapter.generate(
                    prompts[index],
                    seeds[index],
                    self._config.num_inference_steps,
                    self._config.guidance_scale,
                )
                images[index] = image
                path = self._cache_path(adapter.model_id, prompts[index], seeds[index])
                if path is not None:
                    torch.save(image, path)
        return torch.stack(images)

    # ------------------------------------------------------------------
    # GenerativeValidator protocol
    # ------------------------------------------------------------------

    def validate(
        self,
        model: Any,
        concept: str,
        num_samples: int,
        seed: int,
    ) -> RecoveryResult:
        """Generate held-out concept images + neutral controls; score recovery.

        Args:
            model: a :class:`GenerationAdapter` for the (possibly edited) model.
            concept: concept name; must exist in ``concept_prompts``.
            num_samples: number of held-out concept images to generate. If it
                exceeds the number of held-out prompts, prompts cycle with a
                per-index seed so every image is still distinct and stable.
            seed: run-level seed; per-image seeds derive from it.
        """
        adapter = resolve_adapter(model)
        if concept not in self._concept_prompts:
            raise KeyError(f"No held-out prompts registered for concept {concept!r}.")
        started = time.perf_counter()

        heldout = self._concept_prompts[concept]
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive; got {num_samples}.")
        concept_prompts = [heldout[i % len(heldout)] for i in range(num_samples)]
        concept_images = self.generate_images(adapter, concept_prompts, seed)

        neutral_prompts = self._neutral_prompts[: self._config.neutral_samples]
        neutral_images = self.generate_images(adapter, neutral_prompts, seed)

        confidences = self._classify_batched(concept_images, concept)
        neutral_confidences = self._classify_batched(neutral_images, concept)

        threshold = self._config.recovery_threshold
        recovery_rate = sum(c >= threshold for c in confidences) / len(confidences)
        false_positive_rate = sum(c >= threshold for c in neutral_confidences) / max(
            1, len(neutral_confidences)
        )
        wall_clock = time.perf_counter() - started
        return RecoveryResult(
            concept=concept,
            method=adapter.model_id,
            recovery_score=float(recovery_rate),
            drift_score=float(false_positive_rate),
            metrics={
                "mean_confidence": float(sum(confidences) / len(confidences)),
                "max_confidence": float(max(confidences)),
                "num_samples": float(len(confidences)),
                "neutral_samples": float(len(neutral_confidences)),
                "recovery_threshold": float(threshold),
            },
            cost={
                "wall_clock_sec": float(wall_clock),
                "sample_count": int(len(confidences) + len(neutral_confidences)),
                "num_inference_steps": int(self._config.num_inference_steps),
            },
        )

    def _classify_batched(self, images: Tensor, concept: str) -> List[float]:
        scores: List[float] = []
        for start in range(0, images.shape[0], self._config.batch_size):
            batch = images[start : start + self._config.batch_size]
            scores.extend(float(s) for s in self._classifier.classify(batch, concept))
        return scores


# ----------------------------------------------------------------------
# Paired generation and gap helpers
# ----------------------------------------------------------------------


def generate_paired_images(
    adapters: Mapping[str, GenerationAdapter],
    prompts: Sequence[str],
    seed: int,
    config: ValidatorConfig = ValidatorConfig(),
) -> Dict[str, Tensor]:
    """Generate one image per prompt for each adapter with *paired* seeds.

    The pairing is enforced at the seed level: image ``i`` for model A and
    model B is generated from the same prompt and the same run-level seed
    (with per-model seed derivation), which is what makes FID/LPIPS
    comparisons paired rather than independent samples.
    """
    return {
        name: _generate_with_config(adapter, prompts, seed, config)
        for name, adapter in adapters.items()
    }


def _generate_with_config(
    adapter: GenerationAdapter,
    prompts: Sequence[str],
    seed: int,
    config: ValidatorConfig,
) -> Tensor:
    validator = DeterministicGenerativeValidator(
        classifier=_NullClassifier(),
        concept_prompts={"_": prompts},
        neutral_prompts=prompts,
        config=config,
    )
    return validator.generate_images(adapter, prompts, seed)


class _NullClassifier:
    def classify(self, images: Tensor | Sequence[Any], concept: str) -> Sequence[float]:
        return [0.0] * (images.shape[0] if isinstance(images, Tensor) else len(images))


def recovery_gap(
    base_result: RecoveryResult,
    edited_result: RecoveryResult,
) -> float:
    """Erased-vs-unerased recovery gap (base minus edited), in ``[0, 1]``."""
    return float(base_result.recovery_score - edited_result.recovery_score)


def result_to_dict(result: RecoveryResult) -> Dict[str, Any]:
    """JSON-safe serialization of a ``RecoveryResult``."""
    return {
        "concept": result.concept,
        "method": result.method,
        "recovery_score": result.recovery_score,
        "drift_score": result.drift_score,
        "metrics": dict(result.metrics),
        "cost": dict(result.cost),
    }
