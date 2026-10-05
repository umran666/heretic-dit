"""Unit tests for DiffusersModelAdapter."""

import pytest
import torch
import torch.nn as nn

from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter, TargetLayerInfo
from heretic_dit.interfaces import EditSpec, NoisePredictor


class MockCrossAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Self-attention layers (attn1)
        self.attn1_to_q = nn.Linear(in_dim, out_dim, bias=False)
        self.attn1_to_k = nn.Linear(in_dim, out_dim, bias=False)
        self.attn1_to_v = nn.Linear(in_dim, out_dim, bias=False)
        # Cross-attention layers (attn2)
        self.attn2_to_q = nn.Linear(in_dim, out_dim, bias=False)
        self.attn2_to_k = nn.Linear(in_dim, out_dim, bias=False)
        self.attn2_to_v = nn.Linear(in_dim, out_dim, bias=False)


class MockDiffusionModel(nn.Module):
    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.block1 = MockCrossAttentionBlock(in_dim, out_dim)
        self.block2 = MockCrossAttentionBlock(in_dim, out_dim)

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor, conditioning: torch.Tensor):
        # Mock forward pass returning tensor matching latents
        return latents * 0.5 + conditioning.sum() * 0.01


def test_protocol_conformance():
    model = MockDiffusionModel()
    adapter = DiffusersModelAdapter(model)
    assert isinstance(adapter, NoisePredictor)


def test_layer_discovery():
    model = MockDiffusionModel(in_dim=32, out_dim=32)
    adapter = DiffusersModelAdapter(model)

    cross_layers = adapter.get_cross_attention_layer_names(layer_types=("key", "value"))
    assert len(cross_layers) == 4
    assert any("block1.attn2_to_k" in name for name in cross_layers)
    assert any("block1.attn2_to_v" in name for name in cross_layers)
    assert any("block2.attn2_to_k" in name for name in cross_layers)
    assert any("block2.attn2_to_v" in name for name in cross_layers)


def test_temporary_edit_rollback():
    torch.manual_seed(42)
    model = MockDiffusionModel(in_dim=32, out_dim=32)
    adapter = DiffusersModelAdapter(model)

    target_layer = "block1.attn2_to_k"
    orig_weight = model.block1.attn2_to_k.weight.clone()

    # Create dummy subspace direction
    direction = torch.randn(32, 1)
    direction = direction / direction.norm()

    spec = EditSpec(
        layers=[target_layer],
        alpha=1.0,
        mode="orthogonal",
        side="input",
        subspace=direction,
    )

    with adapter.temporary_edit(spec):
        edited_weight = model.block1.attn2_to_k.weight
        # Weight must have changed
        assert not torch.allclose(orig_weight, edited_weight)
        # Check projection: W * v should be 0
        projection = edited_weight @ direction
        assert torch.allclose(projection, torch.zeros_like(projection), atol=1e-5)

    # After context exit, weight must be exactly restored
    assert torch.equal(model.block1.attn2_to_k.weight, orig_weight)


def test_temporary_edit_exception_safety():
    model = MockDiffusionModel(in_dim=16, out_dim=16)
    adapter = DiffusersModelAdapter(model)
    orig_weight = model.block1.attn2_to_k.weight.clone()

    direction = torch.randn(16, 1)
    direction = direction / direction.norm()
    spec = EditSpec(layers=["block1.attn2_to_k"], subspace=direction)

    with pytest.raises(RuntimeError):
        with adapter.temporary_edit(spec):
            raise RuntimeError("Simulation error during Optuna trial")

    # Ensure rollback occurred even though exception was raised
    assert torch.equal(model.block1.attn2_to_k.weight, orig_weight)


def test_predict_noise():
    model = MockDiffusionModel(in_dim=8, out_dim=8)
    adapter = DiffusersModelAdapter(model)

    latents = torch.randn(2, 4, 8, 8)
    timesteps = torch.tensor([500, 500])
    conditioning = torch.randn(2, 77, 8)

    noise_pred = adapter.predict_noise(latents, timesteps, conditioning)
    assert noise_pred.shape == latents.shape
