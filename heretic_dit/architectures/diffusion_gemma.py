"""Adapter for Google DiffusionGemma non-autoregressive block-diffusion models.

Handles:
- 3D batched MoE expert parameter tensors (e.g. [128, 2816, 704]).
- Memory-aliasing and tied encoder-decoder weights (preventing double-projection via data_ptr).
- Router gate protection (preventing router logit flipping and expert collapse).
- Forward hooks for canvas hidden-state extraction without relying on output_hidden_states.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from heretic_dit.core.projector import project_weights, project_weights_
from heretic_dit.interfaces import EditSpec


@dataclass
class MoELayerInfo:
    """Metadata for an abliterable DiffusionGemma MoE or Linear layer."""
    name: str
    parameter: nn.Parameter
    shape: Tuple[int, ...]
    is_batched_moe: bool
    num_experts: int
    data_ptr: int
    category: str  # 'moe_expert', 'attention', 'router_gate', or 'other'


class DiffusionGemmaAdapter:
    """Specialized adapter for DiffusionGemma block-diffusion architectures."""

    def __init__(self, model: nn.Module):
        """Initialize DiffusionGemma adapter.
        
        Args:
            model: The DiffusionGemma model instance.
        """
        self.model = model
        # Route to inner language model if wrapped inside custom encoder/decoder classes
        self.root_module = self._resolve_root_module(model)
        self.layers: Dict[str, MoELayerInfo] = self._discover_layers()

    def _resolve_root_module(self, model: nn.Module) -> nn.Module:
        """Resolve model root, unwrapping wrapper classes if present."""
        if hasattr(model, "encoder") and hasattr(model.encoder, "language_model"):
            return model.encoder.language_model
        if hasattr(model, "language_model"):
            return model.language_model
        return model

    def _discover_layers(self) -> Dict[str, MoELayerInfo]:
        """Discover all abliterable 2D linear and 3D batched MoE layers."""
        discovered: Dict[str, MoELayerInfo] = {}

        for name, module in self.root_module.named_modules():
            # Check for direct Parameter attributes (including custom 3D batched MoE modules)
            for param_name, param in module.named_parameters(recurse=False):
                if not isinstance(param, nn.Parameter) or param.dtype not in (
                    torch.float16, torch.bfloat16, torch.float32, torch.float64
                ):
                    continue

                full_name = f"{name}.{param_name}" if name else param_name
                lower_name = full_name.lower()

                # Determine category
                category = "other"
                if "gate" in lower_name or "router" in lower_name:
                    category = "router_gate"
                elif any(attn in lower_name for attn in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"]):
                    category = "attention"
                elif "expert" in lower_name or "moe" in lower_name or "mlp" in lower_name:
                    category = "moe_expert"

                is_batched_moe = (param.ndim == 3)
                num_experts = param.shape[0] if is_batched_moe else 1

                discovered[full_name] = MoELayerInfo(
                    name=full_name,
                    parameter=param,
                    shape=tuple(param.shape),
                    is_batched_moe=is_batched_moe,
                    num_experts=num_experts,
                    data_ptr=param.data_ptr(),
                    category=category,
                )

        return discovered

    def get_layer_names(
        self,
        include_moe: bool = True,
        include_attention: bool = True,
        protect_router_gates: bool = True,
    ) -> List[str]:
        """Return layer names matching the desired architectural criteria."""
        results = []
        for name, info in self.layers.items():
            if info.category == "router_gate":
                if not protect_router_gates:
                    results.append(name)
                continue
            if info.category == "moe_expert" and include_moe:
                results.append(name)
            elif info.category == "attention" and include_attention:
                results.append(name)
        return results

    def apply_edit(self, spec: EditSpec) -> Dict[str, Tensor]:
        """Apply EditSpec with memory-aliasing deduplication.
        
        Guarantees that tied encoder-decoder parameters (sharing the same data_ptr)
        are modified exactly once, preventing double-projection or delta accumulation.
        
        Args:
            spec: The EditSpec defining target layers, alpha, and subspace.
            
        Returns:
            Dictionary mapping layer names to their original detached clone weights.
        """
        if spec.subspace is None:
            raise ValueError("EditSpec.subspace cannot be None.")

        saved_weights: Dict[str, Tensor] = {}
        processed_data_ptrs: Set[int] = set()

        for layer_name in spec.layers:
            if layer_name not in self.layers:
                continue

            info = self.layers[layer_name]
            param = info.parameter

            # Memory safety: Skip if this underlying memory buffer was already projected
            if info.data_ptr in processed_data_ptrs:
                continue

            # Save clean clone of original weight for exact rollback
            saved_weights[layer_name] = param.detach().clone()
            processed_data_ptrs.add(info.data_ptr)

            # Apply projection (supports both 2D Linear and 3D batched MoE tensors)
            if spec.alpha == 1.0:
                project_weights_(
                    param,
                    spec.subspace,
                    mode=spec.mode,
                    side=spec.side,
                    regularization=spec.regularization,
                )
            else:
                projected = project_weights(
                    param,
                    spec.subspace,
                    mode=spec.mode,
                    side=spec.side,
                    regularization=spec.regularization,
                )
                with torch.no_grad():
                    param.copy_((1.0 - spec.alpha) * param + spec.alpha * projected)

        return saved_weights

    def rollback_edit(self, saved_weights: Dict[str, Tensor]) -> None:
        """Restore weights saved by apply_edit with memory deduplication."""
        restored_ptrs: Set[int] = set()
        with torch.no_grad():
            for layer_name, orig_tensor in saved_weights.items():
                if layer_name in self.layers:
                    info = self.layers[layer_name]
                    if info.data_ptr not in restored_ptrs:
                        info.parameter.copy_(orig_tensor)
                        restored_ptrs.add(info.data_ptr)

    @contextlib.contextmanager
    def temporary_edit(self, spec: EditSpec) -> Generator[DiffusionGemmaAdapter, None, None]:
        """Context manager applying an edit temporarily and guaranteeing restoration."""
        saved_weights = self.apply_edit(spec)
        try:
            yield self
        finally:
            self.rollback_edit(saved_weights)

    @contextlib.contextmanager
    def capture_hidden_states(
        self, layer_names: Sequence[str]
    ) -> Generator[Dict[str, List[Tensor]], None, None]:
        """Capture intermediate activations on target layers using PyTorch forward hooks.
        
        Yields:
            Dictionary mapping layer_name to a list of captured activation tensors.
        """
        captured: Dict[str, List[Tensor]] = {name: [] for name in layer_names}
        hooks = []

        def make_hook(name: str):
            def hook_fn(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
                if isinstance(output, Tensor):
                    captured[name].append(output.detach())
                elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Tensor):
                    captured[name].append(output[0].detach())
            return hook_fn

        for name, module in self.root_module.named_modules():
            if name in captured:
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            yield captured
        finally:
            for h in hooks:
                h.remove()
