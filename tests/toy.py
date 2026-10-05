"""Tiny CPU-only stand-ins for a UNet and a diffusion transformer.

Everything here is random, deterministic and download-free so the drift and
search tests run anywhere.  The toy modules expose real cross-attention-shaped
``to_q`` / ``to_k`` / ``to_v`` / ``to_out`` ``nn.Linear`` layers so the same
target-discovery code paths as a real model are exercised.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

__all__ = ["ToyAttentionBlock", "ToyUNet", "ToyDiT", "OrderedOutput"]


class OrderedOutput:
    """Stand-in for a diffusers output object exposing ``.sample``."""

    __slots__ = ("sample",)

    def __init__(self, sample: Tensor) -> None:
        self.sample = sample


class ToyAttentionBlock(nn.Module):
    """Cross-attention projection block with no actual attention math."""

    def __init__(self, width: int = 8, context: int = 5) -> None:
        super().__init__()
        self.to_q = nn.Linear(width, width)
        self.to_k = nn.Linear(context, width)
        self.to_v = nn.Linear(context, width)
        self.to_out = nn.Linear(width, width)

    def forward(self, latents: Tensor, context: Tensor) -> Tensor:
        query = self.to_q(latents)
        key = self.to_k(context)
        value = self.to_v(context)
        scale = key.shape[-1] ** -0.5
        scores = torch.einsum("bnd,bmd->bnm", query, key) * scale
        weights = scores.softmax(dim=-1)
        attended = torch.einsum("bnm,bmd->bnd", weights, value)
        return latents + self.to_out(attended)


class ToyUNet(nn.Module):
    """UNet2DConditionModel-like backbone returning ``.sample``."""

    def __init__(self, width: int = 8, context: int = 5, depth: int = 3) -> None:
        super().__init__()
        self.width = width
        self.blocks = nn.ModuleList(ToyAttentionBlock(width, context) for _ in range(depth))

    def forward(self, latents: Tensor, timesteps: Tensor, encoder_hidden_states: Tensor) -> OrderedOutput:
        hidden = latents
        context = encoder_hidden_states
        for block in self.blocks:
            hidden = block(hidden, context)
        return OrderedOutput(hidden)


class ToyDiT(nn.Module):
    """SD3 / Flux-like transformer taking hidden_states + timestep + pooled."""

    def __init__(self, width: int = 8, context: int = 5, pooled: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.width = width
        self.blocks = nn.ModuleList(ToyAttentionBlock(width, context) for _ in range(depth))
        self.pooled_proj = nn.Linear(pooled, width)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        pooled_projections: Optional[Tensor] = None,
        **extra: Any,
    ) -> OrderedOutput:
        del extra
        hidden = hidden_states
        for block in self.blocks:
            hidden = block(hidden, encoder_hidden_states)
        if pooled_projections is not None:
            hidden = hidden + self.pooled_proj(pooled_projections).unsqueeze(1)
        return OrderedOutput(hidden)


def toy_conditioning(
    batch: int, context: int = 5, pooled: int = 4, seed: int = 0, device: Optional[torch.device] = None
) -> Dict[str, Tensor]:
    """Deterministic conditioning mapping for :class:`ToyDiT`."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    encoder = torch.randn((batch, 4, context), generator=generator)
    pooled_tensor = torch.randn((batch, pooled), generator=generator)
    if device is not None:
        encoder, pooled_tensor = encoder.to(device), pooled_tensor.to(device)
    return {"encoder_hidden_states": encoder, "pooled_projections": pooled_tensor}


def toy_latents(batch: int = 6, sequence: int = 4, width: int = 8, seed: int = 0) -> Tensor:
    """Deterministic clean latents ``(B, S, width)``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((batch, sequence, width), generator=generator)