"""Pluggable training backends for the recovery baselines.

Real training (diffusers UNet loops, peft LoRA) lives behind lazy imports so
the core test suite runs without heavy dependencies. Tests inject tiny
stand-in backends that satisfy the same protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, runtime_checkable

__all__ = ["TrainingOutcome", "TextualInversionBackend", "FinetuneBackend"]


@dataclass
class TrainingOutcome:
    """Result of a baseline training run.

    Attributes:
        recovered_model: the model after training (new object; backends must
            not mutate ``erased_model`` in place).
        trainable_params: number of parameters actually trained.
        sample_count: number of training images/examples consumed.
        peak_vram_mb: peak GPU memory during training (0 on CPU).
        notes: backend-specific provenance (rank, embedding dim, ...).
    """

    recovered_model: Any
    trainable_params: int
    sample_count: int
    peak_vram_mb: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TextualInversionBackend(Protocol):
    """Protocol for textual-inversion-style pseudo-token training."""

    def train_token(
        self,
        model: Any,
        concept: str,
        prompts: Sequence[str],
        steps: int,
        lr: float,
        seed: int,
    ) -> TrainingOutcome:
        """Learn a pseudo-token for ``concept``; return the patched model."""
        ...


@runtime_checkable
class FinetuneBackend(Protocol):
    """Protocol for LoRA or full fine-tuning recovery."""

    def train(
        self,
        model: Any,
        concept: str,
        prompts: Sequence[str],
        steps: int,
        lr: float,
        rank: int,
        seed: int,
    ) -> TrainingOutcome:
        """Fine-tune ``model`` toward recovering ``concept``; return outcome."""
        ...
