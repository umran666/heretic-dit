"""Core protocol interfaces and data contracts for Heretic-DiT.

All modules (core, search, eval, baselines, architectures) program against
these protocols to maintain clean separation of concerns across agent boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Data Specifications & Results
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EditSpec:
    """Specification of an abliteration / subspace edit applied to model weights.
    
    Attributes:
        layers: Target module paths or layer identifiers to modify.
        alpha: Global or per-layer scaling multiplier for the projection update.
        mode: Projection mode ('orthogonal' or 'covariance').
        side: Target projection side ('input' for W P or 'output' for P W).
        subspace: Optional explicit projection basis (d, k) tensor.
        regularization: Shrinkage parameter for covariance inversion.
        metadata: Arbitrary diagnostic or provenance info (e.g., source concept).
    """
    layers: Sequence[str]
    alpha: float = 1.0
    mode: Literal["orthogonal", "covariance"] = "orthogonal"
    side: Literal["input", "output"] = "input"
    subspace: Optional[Tensor] = None
    regularization: float = 1e-4
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Evaluation result for concept recovery and quality metrics.
    
    Attributes:
        concept: The target concept evaluated.
        method: The method or baseline evaluated (e.g., 'heretic-dit', 'textual-inversion').
        recovery_score: Primary recovery metric (e.g., classifier accuracy or CLIP score gap).
        drift_score: Collateral damage / drift on neutral/control distribution.
        metrics: Detailed metrics breakdown (e.g., FID, LPIPS, CLIP prompt alignment).
        cost: Resource cost accounting (wall_clock_sec, peak_vram_mb, trainable_params, sample_count).
    """
    concept: str
    method: str
    recovery_score: float
    drift_score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyEvaluation:
    """Fast proxy metrics evaluated during Optuna search."""
    proxy_recovery: float
    proxy_drift: float
    trial_id: Optional[int] = None
    edit_spec: Optional[EditSpec] = None


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------

@runtime_checkable
class SubspaceProvider(Protocol):
    """Protocol for extracting or providing concept and neutral subspaces."""

    def get_subspace(
        self,
        concept: str,
        layer_name: str,
        dim: int,
    ) -> Tensor:
        """Return an orthonormal basis tensor of shape (d, k) for the concept."""
        ...

    def get_covariance(
        self,
        layer_name: str,
        dim: int,
    ) -> Optional[Tensor]:
        """Return neutral covariance tensor of shape (d, d), or None if unused."""
        ...


@runtime_checkable
class NoisePredictor(Protocol):
    """Protocol for evaluating diffusion noise predictions across timesteps."""

    def predict_noise(
        self,
        latents: Tensor,
        timesteps: Tensor,
        conditioning: Any,
    ) -> Tensor:
        """Predict noise epsilon_theta(x_t, t, c) for given noisy latents."""
        ...


@runtime_checkable
class ConceptScorer(Protocol):
    """Protocol for scoring concept presence from embeddings or latents."""

    def score(self, representation: Tensor, concept: str) -> float:
        """Return a scalar alignment / recovery score for the given representation."""
        ...


@runtime_checkable
class ConceptClassifier(Protocol):
    """Protocol for classifying generated images to detect target concepts."""

    def classify(self, images: Tensor | Sequence[Any], concept: str) -> Sequence[float]:
        """Return per-image confidence scores in [0.0, 1.0] for the target concept."""
        ...


@runtime_checkable
class GenerativeValidator(Protocol):
    """Protocol for generating images with fixed seeds and scoring recovery."""

    def validate(
        self,
        model: Any,
        concept: str,
        num_samples: int,
        seed: int,
    ) -> RecoveryResult:
        """Run generative evaluation and compute ground-truth recovery and quality."""
        ...


@runtime_checkable
class BaselineRunner(Protocol):
    """Protocol for benchmark baselines (e.g. Textual Inversion, LoRA, Null edit)."""

    def run(
        self,
        erased_model: Any,
        concept: str,
        budget: Dict[str, Any],
    ) -> RecoveryResult:
        """Execute baseline recovery procedure under the given compute budget."""
        ...
