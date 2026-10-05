"""High-level unified pipeline for Heretic-DiT.

Wires together subspace extraction, noise cache construction,
drift evaluation, and model parameter surgery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
from heretic_dit.core.projector import project_weights, project_weights_
from heretic_dit.core.subspace import (
    contrastive_pca,
    mean_difference,
    subspace_svd,
)
from heretic_dit.interfaces import EditSpec, NoisePredictor, ProxyEvaluation
from heretic_dit.metrics.drift import NoiseCache, compute_epsilon_drift


@dataclass
class AuditReport:
    """Diagnostic audit report evaluating an abliteration edit."""
    concept: str
    target_layers: List[str]
    alpha: float
    projection_mode: str
    subspace_rank: int
    epsilon_drift: float
    wall_clock_sec: float
    metadata: Dict[str, Any]


class HereticDiTPipeline:
    """Unified pipeline for auditing and applying subspace abliteration on diffusion models."""

    def __init__(
        self,
        adapter: Union[DiffusersModelAdapter, NoisePredictor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize HereticDiT pipeline.
        
        Args:
            adapter: Model adapter wrapping the diffusion backbone.
            device: Execution device for evaluation.
            dtype: Floating point precision for computations.
        """
        self.adapter = adapter
        self.device = device
        self.dtype = dtype

    def extract_concept_direction(
        self,
        target_activations: Tensor,
        neutral_activations: Optional[Tensor] = None,
        method: Literal["mean_diff", "contrastive_pca", "svd"] = "mean_diff",
        max_rank: int = 1,
    ) -> Tensor:
        """Extract orthonormal concept basis V from activation tensors.
        
        Args:
            target_activations: Activations representing target concept / refusal.
            neutral_activations: Activations on neutral / harmless control prompts.
            method: Subspace extraction method ('mean_diff', 'contrastive_pca', 'svd').
            max_rank: Maximum subspace rank (dimension k).
            
        Returns:
            Orthonormal basis tensor of shape (d, k).
        """
        if method == "mean_diff":
            if neutral_activations is None:
                raise ValueError("neutral_activations required for mean_diff")
            basis = mean_difference(target_activations, neutral_activations)
        elif method == "contrastive_pca":
            if neutral_activations is None:
                raise ValueError("neutral_activations required for contrastive_pca")
            basis = contrastive_pca(target_activations, neutral_activations, k=max_rank)
        elif method == "svd":
            basis = subspace_svd(target_activations, max_rank=max_rank)
        else:
            raise ValueError(f"Unknown subspace extraction method: {method}")

        return basis.to(device=self.device, dtype=self.dtype)

    def evaluate_drift(
        self,
        noise_cache: NoiseCache,
        spec: EditSpec,
    ) -> float:
        """Evaluate epsilon-prediction drift on neutral prompts under an EditSpec.
        
        Uses the adapter's temporary_edit context manager to test the edit reversibly.
        """
        if hasattr(self.adapter, "temporary_edit"):
            with self.adapter.temporary_edit(spec):
                drift_value = compute_epsilon_drift(
                    base_model=self.adapter,
                    edited_model=self.adapter,
                    cache=noise_cache,
                )
                if isinstance(drift_value, (float, int)):
                    return float(drift_value)
                elif hasattr(drift_value, "item"):
                    return float(drift_value.item())
                elif isinstance(drift_value, dict):
                    return float(drift_value.get("mean_drift", list(drift_value.values())[0]))
                return float(drift_value)
        else:
            raise NotImplementedError("Adapter must implement temporary_edit to evaluate drift.")

    def apply_permanent_edit(self, spec: EditSpec) -> None:
        """Permanently apply the abliteration edit to model weights."""
        if hasattr(self.adapter, "apply_edit"):
            self.adapter.apply_edit(spec)
        else:
            raise NotImplementedError("Adapter must implement apply_edit for permanent modification.")
