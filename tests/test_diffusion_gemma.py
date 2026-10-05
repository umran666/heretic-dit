"""Unit tests for DiffusionGemmaAdapter with batched MoE tensors and tied memory."""

import pytest
import torch
import torch.nn as nn

from heretic_dit.architectures.diffusion_gemma import DiffusionGemmaAdapter
from heretic_dit.interfaces import EditSpec


class MockMoEBlock(nn.Module):
    def __init__(self, num_experts: int = 4, in_dim: int = 16, out_dim: int = 16):
        super().__init__()
        # 3D batched expert weights: [num_experts, out_dim, in_dim]
        self.experts_weight = nn.Parameter(torch.randn(num_experts, out_dim, in_dim))
        self.router_gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple forward mock
        return x @ self.experts_weight[0].T


class MockTiedEncoderDecoder(nn.Module):
    def __init__(self, in_dim: int = 16):
        super().__init__()
        self.moe_block = MockMoEBlock(num_experts=4, in_dim=in_dim, out_dim=in_dim)
        # Tied weight: decoder references exact same parameter tensor (same data_ptr)
        self.decoder_moe_weight = self.moe_block.experts_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe_block(x)


def test_moe_layer_discovery():
    model = MockTiedEncoderDecoder(in_dim=16)
    adapter = DiffusionGemmaAdapter(model)

    assert "moe_block.experts_weight" in adapter.layers
    info = adapter.layers["moe_block.experts_weight"]
    assert info.is_batched_moe is True
    assert info.num_experts == 4
    assert info.category == "moe_expert"


def test_router_gate_protection():
    model = MockTiedEncoderDecoder(in_dim=16)
    adapter = DiffusionGemmaAdapter(model)

    # When protect_router_gates=True, router gate should not be included
    protected_layers = adapter.get_layer_names(protect_router_gates=True)
    assert not any("router_gate" in name for name in protected_layers)

    # When protect_router_gates=False, router gate should be included if requested
    unprotected_layers = adapter.get_layer_names(protect_router_gates=False)
    assert any("router_gate" in name for name in unprotected_layers)


def test_tied_memory_deduplication():
    model = MockTiedEncoderDecoder(in_dim=16)
    adapter = DiffusionGemmaAdapter(model)

    # Verify both attributes share the exact same data_ptr
    ptr1 = model.moe_block.experts_weight.data_ptr()
    ptr2 = model.decoder_moe_weight.data_ptr()
    assert ptr1 == ptr2

    # Target both tied layer names in spec
    direction = torch.randn(16, 1)
    direction = direction / direction.norm()

    spec = EditSpec(
        layers=["moe_block.experts_weight", "decoder_moe_weight"],
        alpha=1.0,
        mode="orthogonal",
        side="input",
        subspace=direction,
    )

    orig_copy = model.moe_block.experts_weight.clone()

    with adapter.temporary_edit(spec):
        edited = model.moe_block.experts_weight
        # Check projection: W[e] @ v should be zero for each expert e
        for e in range(4):
            proj = edited[e] @ direction
            assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-5)

    # After temporary edit, weights must be bitwise restored
    assert torch.equal(model.moe_block.experts_weight, orig_copy)
    assert torch.equal(model.decoder_moe_weight, orig_copy)


def test_capture_hidden_states():
    model = MockTiedEncoderDecoder(in_dim=16)
    adapter = DiffusionGemmaAdapter(model)

    x = torch.randn(2, 5, 16)
    with adapter.capture_hidden_states(["moe_block"]) as captured:
        _ = model(x)

    assert "moe_block" in captured
    assert len(captured["moe_block"]) == 1
    assert captured["moe_block"][0].shape == x.shape
