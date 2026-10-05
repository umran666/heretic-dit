"""Diffusers model adapter for cross-attention layer discovery, activation extraction,
and reversible weight projections on UNet and DiT architectures.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from heretic_dit.core.projector import ProjectionMode, ProjectionSide, project_weights, project_weights_
from heretic_dit.interfaces import EditSpec, NoisePredictor


@dataclass
class TargetLayerInfo:
    """Metadata for an abliterable cross-attention layer."""
    name: str
    module: nn.Module
    weight_shape: Tuple[int, ...]
    is_cross_attention: bool
    layer_type: str  # 'key', 'value', 'query', or 'other'


class DiffusersModelAdapter(NoisePredictor):
    """Adapter for Diffusers UNet and Diffusion Transformer (DiT) models.
    
    Provides:
    - Automatic discovery of cross-attention key and value projection weights.
    - Activation capture during diffusion inference via forward hooks.
    - Safe in-place parameter editing with context-managed rollbacks.
    - Implementation of the `NoisePredictor` protocol for epsilon-drift evaluation.
    """

    def __init__(self, model: nn.Module, is_dit: bool = False):
        """Initialize adapter.
        
        Args:
            model: The denoising backbone (UNet2DConditionModel or DiT transformer).
            is_dit: True if the model is a Diffusion Transformer (e.g. Flux, SD3, PixArt).
        """
        super().__init__()
        self.model = model
        self.is_dit = is_dit
        self.target_layers: Dict[str, TargetLayerInfo] = self._discover_target_layers()

    def _discover_target_layers(self) -> Dict[str, TargetLayerInfo]:
        """Discover cross-attention key and value projection layers."""
        targets: Dict[str, TargetLayerInfo] = {}

        for name, module in self.model.named_modules():
            # Check for standard cross-attention projection layers
            # In UNet: attn2 is cross-attention (attn1 is self-attention)
            # In DiT: joint_blocks or cross_attn modules
            if not hasattr(module, "weight") or not isinstance(module.weight, (torch.nn.Parameter, Tensor)):
                continue

            lower_name = name.lower()
            is_cross = False
            layer_type = "other"

            # Detect key and value projections
            if any(k in lower_name for k in ["to_k", "k_proj", "key"]):
                layer_type = "key"
            elif any(v in lower_name for v in ["to_v", "v_proj", "value"]):
                layer_type = "value"
            elif any(q in lower_name for q in ["to_q", "q_proj", "query"]):
                layer_type = "query"

            if layer_type in ("key", "value"):
                # Determine if it is cross-attention
                if "attn2" in lower_name or "cross" in lower_name or "context" in lower_name:
                    is_cross = True
                elif self.is_dit:
                    # In DiT, attention blocks typically process joint text-image tokens
                    is_cross = True

                targets[name] = TargetLayerInfo(
                    name=name,
                    module=module,
                    weight_shape=tuple(module.weight.shape),
                    is_cross_attention=is_cross,
                    layer_type=layer_type,
                )

        return targets

    def get_cross_attention_layer_names(self, layer_types: Tuple[str, ...] = ("key", "value")) -> List[str]:
        """Return names of discovered cross-attention layers matching target types."""
        return [
            name for name, info in self.target_layers.items()
            if info.is_cross_attention and info.layer_type in layer_types
        ]

    def predict_noise(
        self,
        latents: Tensor,
        timesteps: Tensor,
        conditioning: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Evaluate model noise prediction epsilon_theta(x_t, t, c).
        
        Implements the `NoisePredictor` protocol.
        """
        # Call model forward pass
        output = self.model(latents, timesteps, conditioning, **kwargs)
        # Handle standard diffusers output dataclass (e.g., UNet2DConditionOutput.sample)
        if hasattr(output, "sample"):
            return output.sample
        elif isinstance(output, tuple):
            return output[0]
        return output

    def apply_edit(self, spec: EditSpec) -> Dict[str, Tensor]:
        """Apply an EditSpec to the target layers, returning original weights for rollback.
        
        Args:
            spec: The EditSpec defining target layers, alpha, mode, and subspace.
            
        Returns:
            Dictionary mapping layer names to their original detached weight tensors.
        """
        if spec.subspace is None:
            raise ValueError("EditSpec.subspace must not be None to apply an edit.")

        saved_weights: Dict[str, Tensor] = {}
        subspace = spec.subspace

        for layer_name in spec.layers:
            if layer_name not in self.target_layers:
                continue

            module = self.target_layers[layer_name].module
            weight = module.weight

            # Save clean clone of original weight
            saved_weights[layer_name] = weight.detach().clone()

            # Apply projection update with alpha scaling
            # W* = (1 - alpha) * W + alpha * Project(W)
            if spec.alpha == 0.0:
                continue
            elif spec.alpha == 1.0:
                project_weights_(
                    weight,
                    subspace,
                    mode=spec.mode,
                    side=spec.side,
                    regularization=spec.regularization,
                )
            else:
                projected = project_weights(
                    weight,
                    subspace,
                    mode=spec.mode,
                    side=spec.side,
                    regularization=spec.regularization,
                )
                with torch.no_grad():
                    weight.copy_((1.0 - spec.alpha) * weight + spec.alpha * projected)

        return saved_weights

    def rollback_edit(self, saved_weights: Dict[str, Tensor]) -> None:
        """Restore original weights saved by `apply_edit`."""
        with torch.no_grad():
            for layer_name, orig_weight in saved_weights.items():
                if layer_name in self.target_layers:
                    self.target_layers[layer_name].module.weight.copy_(orig_weight)

    @contextlib.contextmanager
    def temporary_edit(self, spec: EditSpec) -> Generator[DiffusersModelAdapter, None, None]:
        """Context manager applying an edit temporarily and guaranteeing restoration."""
        saved_weights = self.apply_edit(spec)
        try:
            yield self
        finally:
            self.rollback_edit(saved_weights)
