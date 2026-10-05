"""Tests for prediction-drift metrics and the predictor adapters (CPU toy models)."""

import pytest
import torch

from heretic_dit.metrics.drift import (
    DiTPredictor,
    NoiseCache,
    UNetPredictor,
    add_noise,
    compute_epsilon_drift,
    compute_prediction_drift,
    linear_beta_schedule,
    stratified_timesteps,
    timestep_bins,
)
from heretic_dit.search.editing import LayerEdit, applied_edit, cross_attention_targets
from tests.toy import ToyDiT, ToyUNet, toy_conditioning, toy_latents

# Toy models are tiny and exact; disable autocast so comparisons are bit-true.
NO_AUTOCAST = None


def _unet_predictor(seed: int = 0) -> UNetPredictor:
    torch.manual_seed(seed)
    return UNetPredictor(ToyUNet(), prediction_type="epsilon")


def _dit_predictor(seed: int = 0) -> DiTPredictor:
    torch.manual_seed(seed)
    return DiTPredictor(ToyDiT(), prediction_type="velocity")


def _unet_cache(predictor, seed: int = 0, batch: int = 6) -> NoiseCache:
    latents = toy_latents(batch=batch, seed=seed)
    cond = torch.randn((batch, 4, 5), generator=torch.Generator().manual_seed(seed + 1))
    return NoiseCache.build(predictor, latents, None, cond, seed=seed, autocast_dtype=NO_AUTOCAST)


def test_identity_edit_has_exactly_zero_drift():
    predictor = _unet_predictor()
    cache = _unet_cache(predictor)
    drift = compute_epsilon_drift(None, predictor, cache=cache, autocast_dtype=NO_AUTOCAST)
    assert drift == 0.0
    breakdown = compute_epsilon_drift(
        None, predictor, cache=cache, return_breakdown=True, autocast_dtype=NO_AUTOCAST
    )
    assert breakdown["drift"] == 0.0
    assert breakdown["relative_drift"] == 0.0
    assert all(value == 0.0 for value in breakdown["per_bin"].values())


def test_alias_matches_compute_epsilon_drift():
    assert compute_prediction_drift is compute_epsilon_drift


@pytest.mark.parametrize("adapter", ["unet", "dit"])
def test_adapters_produce_correct_shapes_and_drift(adapter):
    if adapter == "unet":
        predictor = _unet_predictor()
        latents = toy_latents(batch=4, seed=3)
        cond = torch.randn((4, 4, 5), generator=torch.Generator().manual_seed(4))
    else:
        predictor = _dit_predictor()
        latents = toy_latents(batch=4, seed=3)
        cond = toy_conditioning(4, seed=4)

    cache = NoiseCache.build(predictor, latents, None, cond, seed=7, autocast_dtype=NO_AUTOCAST)
    assert cache.base_pred.shape == latents.shape
    assert cache.prediction_type in ("epsilon", "velocity")

    drift = compute_epsilon_drift(None, predictor, cache=cache, autocast_dtype=NO_AUTOCAST)
    assert drift == 0.0

    # A non-trivial edit must register positive drift under both adapters.
    target = cross_attention_targets(predictor.model)[0]
    dim = target.weight.shape[-1]
    direction = torch.zeros(dim)
    direction[0] = 1.0
    edit = LayerEdit(name=target.name, weight=target.weight, alpha=1.0, directions=direction)
    with applied_edit(predictor.model, [edit]):
        edited = compute_epsilon_drift(None, predictor, cache=cache, autocast_dtype=NO_AUTOCAST)
    assert edited > 0.0


def test_dit_adapter_handles_plain_tensor_and_mapping_conditioning():
    predictor = _dit_predictor(seed=11)
    latents = toy_latents(batch=3, seed=12)
    mapping = toy_conditioning(3, seed=13)
    encoder_only = mapping["encoder_hidden_states"]
    with_mapping = predictor.predict_noise(latents, torch.tensor([500, 501, 502]), mapping)
    with_tensor = predictor.predict_noise(latents, torch.tensor([500, 501, 502]), encoder_only)
    # Pooled projections differ, but both call paths must run and return tensors.
    assert with_mapping.shape == latents.shape and with_tensor.shape == latents.shape


def test_drift_increases_monotonically_with_alpha():
    predictor = _unet_predictor(seed=21)
    cache = _unet_cache(predictor, seed=22)
    targets = cross_attention_targets(predictor.model)
    edits = []
    for index, target in enumerate(targets):
        dim = target.weight.shape[-1]
        generator = torch.Generator().manual_seed(index)
        direction = torch.randn((dim, 1), generator=generator)
        edits.append(
            LayerEdit(name=target.name, weight=target.weight, alpha=0.0, directions=direction)
        )
    drifts = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        resolved = [
            LayerEdit(
                name=edit.name,
                weight=edit.weight,
                alpha=alpha,
                directions=edit.directions,
                mode=edit.mode,
                side=edit.side,
            )
            for edit in edits
        ]
        with applied_edit(predictor.model, resolved):
            drifts.append(
                compute_epsilon_drift(None, predictor, cache=cache, autocast_dtype=NO_AUTOCAST)
            )
    assert drifts[0] == 0.0
    assert all(later > earlier for earlier, later in zip(drifts, drifts[1:])), drifts


def test_cache_and_no_cache_paths_agree():
    import copy

    torch.manual_seed(31)
    base = ToyUNet()
    base_predictor = UNetPredictor(base, prediction_type="epsilon")
    # Cache base predictions from the pristine model, then edit an independent copy.
    latents = toy_latents(batch=6, seed=31)
    cond = torch.randn((6, 4, 5), generator=torch.Generator().manual_seed(32))
    cache = NoiseCache.build(base_predictor, latents, None, cond, seed=33, autocast_dtype=NO_AUTOCAST)

    edited_model = copy.deepcopy(base)
    edited_predictor = UNetPredictor(edited_model, prediction_type="epsilon")
    target = cross_attention_targets(edited_model)[0]
    direction = torch.zeros(target.weight.shape[-1])
    direction[0] = 1.0
    edit = LayerEdit(name=target.name, weight=target.weight, alpha=0.5, directions=direction)

    with applied_edit(edited_model, [edit]):
        cached = compute_epsilon_drift(None, edited_predictor, cache=cache, autocast_dtype=NO_AUTOCAST)
        uncached = compute_epsilon_drift(
            base_predictor,
            edited_predictor,
            cache.x_t,
            cache.t,
            cache.cond,
            autocast_dtype=NO_AUTOCAST,
        )
    assert cached == pytest.approx(uncached, rel=0, abs=0)


def test_applied_edit_restores_bit_exactly_when_body_raises():
    predictor = _unet_predictor(seed=41)
    targets = cross_attention_targets(predictor.model)
    before = {target.name: target.weight.detach().clone() for target in targets}
    pointers = {target.name: target.weight.data_ptr() for target in targets}
    directions = {
        target.name: torch.randn(target.weight.shape[-1]) for target in targets
    }
    edits = [
        LayerEdit(name=t.name, weight=t.weight, alpha=0.75, directions=directions[t.name])
        for t in targets
    ]
    with pytest.raises(RuntimeError, match="boom"):
        with applied_edit(predictor.model, edits):
            raise RuntimeError("boom")
    for target in targets:
        assert target.weight.data_ptr() == pointers[target.name]
        assert torch.equal(target.weight, before[target.name])


def test_applied_edit_restores_when_projector_raises_midway():
    predictor = _unet_predictor(seed=51)
    targets = cross_attention_targets(predictor.model)
    before = [target.weight.detach().clone() for target in targets]
    # Second edit is invalid (non-finite directions) and must abort the first.
    bad = targets[1]
    edits = [
        LayerEdit(
            name=targets[0].name,
            weight=targets[0].weight,
            alpha=0.5,
            directions=torch.ones(targets[0].weight.shape[-1]),
        ),
        LayerEdit(
            name=bad.name,
            weight=bad.weight,
            alpha=0.5,
            directions=torch.full((bad.weight.shape[-1],), float("nan")),
        ),
    ]
    with pytest.raises(ValueError):
        with applied_edit(predictor.model, edits):
            pass
    for target, original in zip(targets, before):
        assert torch.equal(target.weight, original)


def test_return_breakdown_reports_bins_and_relative_drift():
    predictor = _unet_predictor(seed=61)
    latents = toy_latents(batch=9, seed=61)
    cond = torch.randn((9, 4, 5), generator=torch.Generator().manual_seed(62))
    cache = NoiseCache.build(predictor, latents, None, cond, seed=63, autocast_dtype=NO_AUTOCAST)
    assert set(cache.bins.tolist()) == {0, 1, 2}
    target = cross_attention_targets(predictor.model)[0]
    direction = torch.zeros(target.weight.shape[-1])
    direction[0] = 1.0
    edit = LayerEdit(name=target.name, weight=target.weight, alpha=1.0, directions=direction)
    with applied_edit(predictor.model, [edit]):
        breakdown = compute_epsilon_drift(
            None, predictor, cache=cache, return_breakdown=True, autocast_dtype=NO_AUTOCAST
        )
    assert set(breakdown["per_bin"]) == {"low", "mid", "high"}
    assert breakdown["drift"] > 0
    assert breakdown["relative_drift"] > 0
    assert breakdown["num_samples"] == 9
    assert breakdown["prediction_type"] == "epsilon"


def test_cfg_scale_amplifies_drift():
    predictor = _unet_predictor(seed=71)
    latents = toy_latents(batch=4, seed=72)
    cond = torch.randn((4, 4, 5), generator=torch.Generator().manual_seed(73))
    uncond = torch.randn((4, 4, 5), generator=torch.Generator().manual_seed(74))
    cache = NoiseCache.build(
        predictor, latents, None, cond, seed=75, uncond_cond=uncond, autocast_dtype=NO_AUTOCAST
    )
    target = cross_attention_targets(predictor.model)[0]
    direction = torch.zeros(target.weight.shape[-1])
    direction[0] = 1.0
    edit = LayerEdit(name=target.name, weight=target.weight, alpha=0.5, directions=direction)
    with applied_edit(predictor.model, [edit]):
        plain = compute_epsilon_drift(None, predictor, cache=cache, autocast_dtype=NO_AUTOCAST)
        guided = compute_epsilon_drift(
            None, predictor, cache=cache, cfg_scale=7.5, autocast_dtype=NO_AUTOCAST
        )
    assert guided > plain > 0.0


def test_cfg_requires_unconditional_conditioning():
    predictor = _unet_predictor(seed=81)
    cache = _unet_cache(predictor, seed=82)
    with pytest.raises(ValueError, match="cfg_scale requires"):
        compute_epsilon_drift(None, predictor, cache=cache, cfg_scale=5.0, autocast_dtype=NO_AUTOCAST)


def test_cache_requires_base_model_without_cache():
    predictor = _unet_predictor(seed=91)
    with pytest.raises(ValueError, match="base_model is required"):
        compute_epsilon_drift(
            None, predictor, toy_latents(4, seed=1), torch.tensor([1, 2, 3, 4]),
            torch.randn((4, 4, 5)), autocast_dtype=NO_AUTOCAST,
        )


def test_stratified_timesteps_are_deterministic_and_cover_bins():
    first = stratified_timesteps(9, num_bins=3, seed=5)
    second = stratified_timesteps(9, num_bins=3, seed=5)
    assert torch.equal(first, second)
    assert set(timestep_bins(first).tolist()) == {0, 1, 2}
    assert stratified_timesteps(9, num_bins=3, seed=6).tolist() != first.tolist()


def test_add_noise_matches_closed_form():
    alphas = linear_beta_schedule()
    clean = torch.ones((2, 3))
    noise = torch.full((2, 3), 2.0)
    timesteps = torch.tensor([0, 999])
    actual = add_noise(clean, timesteps, noise, alphas)
    ab = alphas[timesteps].reshape(2, 1)
    expected = ab.sqrt() * clean + (1 - ab).sqrt() * noise
    assert actual.dtype == clean.dtype and actual.device == clean.device
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)


def test_invalid_predictor_configuration_rejected():
    with pytest.raises(ValueError, match="prediction_type"):
        UNetPredictor(ToyUNet(), prediction_type="score")
    with pytest.raises(ValueError, match="prediction_type"):
        DiTPredictor(ToyDiT(), prediction_type="score")