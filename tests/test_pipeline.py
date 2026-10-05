"""Unit tests for HereticDiTPipeline end-to-end integration."""

import pytest
import torch

from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
from heretic_dit.interfaces import EditSpec
from heretic_dit.metrics.drift import NoiseCache
from heretic_dit.pipeline import HereticDiTPipeline
from tests.test_diffusers_adapter import MockDiffusionModel


def test_pipeline_subspace_extraction_and_drift():
    torch.manual_seed(42)
    model = MockDiffusionModel(in_dim=32, out_dim=32)
    adapter = DiffusersModelAdapter(model)
    pipeline = HereticDiTPipeline(adapter)

    # 1. Synthesize contrastive activations
    # Target activation has a shift along dim 0
    target_acts = torch.randn(20, 32)
    target_acts[:, 0] += 5.0
    neutral_acts = torch.randn(20, 32)

    basis = pipeline.extract_concept_direction(
        target_activations=target_acts,
        neutral_activations=neutral_acts,
        method="mean_diff",
        max_rank=1,
    )

    assert basis.shape == (32, 1)
    # The basis should be strongly aligned with the shift along dimension 0
    assert abs(float(basis[0, 0])) > 0.8

    # 2. Build synthetic noise cache for drift measurement using NoiseCache.build
    clean_latents = torch.randn(4, 4, 32, 32)
    cond = torch.randn(4, 77, 32)
    cache = NoiseCache.build(
        base_predictor=adapter,
        latents=clean_latents,
        timesteps=torch.tensor([500, 500, 500, 500]),
        cond=cond,
        seed=42,
        autocast_dtype=None,
    )

    # 3. Create EditSpec
    target_layers = adapter.get_cross_attention_layer_names()[:2]
    spec = EditSpec(
        layers=target_layers,
        alpha=0.5,
        mode="orthogonal",
        side="input",
        subspace=basis,
    )

    # 4. Measure drift
    drift = pipeline.evaluate_drift(cache, spec)
    assert isinstance(drift, float)
    assert drift >= 0.0

    # 5. Verify model weights were preserved after evaluate_drift
    # (temporary_edit guarantees restoration)
    current_pred = adapter.predict_noise(cache.x_t, cache.t, cond)
    assert torch.allclose(cache.base_pred, current_pred)
