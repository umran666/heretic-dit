"""Tests for the Optuna objective, editing safety and proxy scorers (CPU toy models)."""

import copy
import math
from typing import Any, Optional

import optuna
import pytest
import torch

from heretic_dit.metrics.drift import NoiseCache, NoisePredictor, UNetPredictor, compute_epsilon_drift
from heretic_dit.search.editing import (
    InterpolationKernel,
    LayerEdit,
    alpha_kernel,
    applied_edit,
    cross_attention_targets,
    resolve_alphas,
)
from heretic_dit.search.optuna_objective import (
    BENIGN_CONCEPT_PROMPTS,
    DenoisingLossScorer,
    DictSubspaceProvider,
    GenerativeValidator,
    ReferenceMatchScorer,
    SubspaceEntry,
    TrialConfig,
    benchmark_trial,
    build_objective,
    create_study,
    resolve_subspace,
)
from tests.toy import ToyUNet, toy_latents

NO_AUTOCAST = None


class FixedPredictor:
    """Predictor returning a preset tensor; used to control MSE exactly."""

    def __init__(self, prediction: torch.Tensor, prediction_type: str = "epsilon") -> None:
        self.prediction = prediction
        self.prediction_type = prediction_type

    def predict_noise(self, latents: torch.Tensor, timesteps: torch.Tensor, conditioning: Any) -> torch.Tensor:
        return self.prediction

    def predict(self, x_t: torch.Tensor, t: torch.Tensor, cond: Any) -> torch.Tensor:
        return self.prediction


class AlphaScorer:
    """Fake ConceptScorer; recovery is a constant in [0, 1] for range checks."""

    def score(self, edited_predictor: NoisePredictor, concept_cache: Optional[NoiseCache] = None) -> float:
        return 0.5


def _model_and_predictor(seed: int = 0, depth: int = 2):
    torch.manual_seed(seed)
    model = ToyUNet(width=8, context=5, depth=depth)
    return model, UNetPredictor(model, prediction_type="epsilon")


def _neutral_cache(predictor, seed: int = 0, batch: int = 6) -> NoiseCache:
    latents = toy_latents(batch=batch, seed=seed)
    cond = torch.randn((batch, 4, 5), generator=torch.Generator().manual_seed(seed + 1))
    return NoiseCache.build(predictor, latents, None, cond, seed=seed, autocast_dtype=NO_AUTOCAST)


def _identity_provider(model: Any, k: int = 1) -> DictSubspaceProvider:
    entries = {}
    for target in cross_attention_targets(model):
        dim = target.weight.shape[-1]
        generator = torch.Generator().manual_seed(abs(hash(target.name)) % 10_000)
        entries[target.name] = SubspaceEntry(directions=torch.randn((dim, k), generator=generator))
    return DictSubspaceProvider(entries)


# -----------------------------------------------------------------------------
# Heretic-style alpha kernel
# -----------------------------------------------------------------------------


def test_alpha_kernel_is_bounded_and_peaks_at_requested_position():
    kernel = InterpolationKernel(alpha_max=2.0, alpha_min=0.1, peak_position=1.0, falloff_distance=0.5)
    values = alpha_kernel(6, kernel)
    assert values.shape == (6,)
    assert values.min() >= 0.1 - 1e-9 and values.max() <= 2.0 + 1e-9
    assert values[-1] == pytest.approx(2.0, abs=1e-9)  # peak at the deepest layer
    assert max(values) == pytest.approx(2.0, abs=1e-9)


def test_alpha_kernel_single_layer_and_validation():
    assert alpha_kernel(1, InterpolationKernel(alpha_max=1.5)).tolist() == [1.5]
    with pytest.raises(ValueError):
        alpha_kernel(4, InterpolationKernel(alpha_max=-1.0))
    with pytest.raises(ValueError):
        alpha_kernel(4, InterpolationKernel(alpha_max=1.0, alpha_min=2.0))
    with pytest.raises(ValueError):
        alpha_kernel(4, InterpolationKernel(alpha_max=1.0, falloff_distance=0.0))
    with pytest.raises(ValueError):
        alpha_kernel(4, InterpolationKernel(alpha_max=1.0, peak_position=1.5))


def test_resolve_alphas_range_and_mask_zero_unselected_layers():
    kernel = InterpolationKernel(alpha_max=1.0, alpha_min=0.0, peak_position=1.0, falloff_distance=1.0)
    range_alphas = resolve_alphas(5, kernel, layer_mode="range", start=1, end=4)
    assert range_alphas[0] == 0.0 and range_alphas[4] == 0.0
    assert all(value > 0 for value in range_alphas[1:4])
    mask_alphas = resolve_alphas(5, kernel, layer_mode="mask", mask=[True, False, True, False, True])
    assert mask_alphas[1] == 0.0 and mask_alphas[3] == 0.0
    assert mask_alphas[0] > 0 and mask_alphas[2] > 0 and mask_alphas[4] > 0
    with pytest.raises(ValueError):
        resolve_alphas(5, kernel, layer_mode="bad")


# -----------------------------------------------------------------------------
# Subspace provider adapter
# -----------------------------------------------------------------------------


def test_resolve_subspace_accepts_spec_get_form():
    direction = torch.randn(8, 2)
    provider = DictSubspaceProvider({"blocks.0.to_k": SubspaceEntry(directions=direction)})
    entry = resolve_subspace(provider, "blocks.0.to_k", 8)
    assert torch.equal(entry.directions, direction)


def test_resolve_subspace_accepts_shared_interfaces_form():
    class SharedProvider:
        def get_subspace(self, concept, layer_name, dim):
            assert dim == 8
            return torch.ones(dim, 1)

        def get_covariance(self, layer_name, dim):
            return torch.eye(dim)

    entry = resolve_subspace(SharedProvider(), "blocks.0.to_v", 8, concept="tench", need_covariance=True)
    assert entry.directions.shape == (8, 1)
    assert entry.covariance is not None and entry.supports_covariance


def test_resolve_subspace_accepts_raw_tensor_and_tuple():
    provider = DictSubspaceProvider({"l": torch.ones(4, 1)})
    assert resolve_subspace(provider, "l", 4).directions.shape == (4, 1)
    provider2 = DictSubspaceProvider({"l": (torch.ones(4, 1), torch.eye(4))})
    assert resolve_subspace(provider2, "l", 4).neutral is not None


# -----------------------------------------------------------------------------
# ReferenceMatchScorer / DenoisingLossScorer
# -----------------------------------------------------------------------------


def test_reference_match_scorer_normalizes_against_erased_baseline():
    generator = torch.Generator().manual_seed(3)
    reference = torch.randn((5, 4), generator=generator)
    erased = reference + 0.5
    cache = NoiseCache(
        x_t=torch.zeros((5, 4)),
        t=torch.zeros(5, dtype=torch.long),
        cond=torch.zeros((5, 4, 5)),
        base_pred=reference,
        bins=torch.zeros(5, dtype=torch.long),
    )
    scorer = ReferenceMatchScorer(reference_predictor=FixedPredictor(reference), concept_cache=cache)
    baseline = scorer.calibrate(FixedPredictor(erased))
    assert baseline > 0
    assert scorer.score(FixedPredictor(reference)) == pytest.approx(1.0, abs=1e-6)
    assert scorer.score(FixedPredictor(erased)) == pytest.approx(0.0, abs=1e-6)
    midpoint = (reference + erased) / 2
    assert scorer.score(FixedPredictor(midpoint)) == pytest.approx(0.75, abs=1e-5)


def test_reference_match_scorer_clips_and_requires_calibration():
    reference = torch.zeros((3, 4))
    cache = NoiseCache(
        x_t=torch.zeros((3, 4)), t=torch.zeros(3, dtype=torch.long),
        cond=torch.zeros((3, 4, 5)), base_pred=reference, bins=torch.zeros(3, dtype=torch.long),
    )
    scorer = ReferenceMatchScorer(reference_predictor=FixedPredictor(reference), concept_cache=cache)
    with pytest.raises(ValueError, match="calibrate"):
        scorer.score(FixedPredictor(reference))
    scorer.calibrate(FixedPredictor(torch.ones((3, 4))))  # baseline = 1.0
    assert scorer.score(FixedPredictor(torch.full((3, 4), 5.0))) == 0.0
    assert 0.0 <= scorer.score(FixedPredictor(reference)) <= 1.0


def test_denoising_loss_scorer_scores_perfect_target_as_one():
    torch.manual_seed(5)
    model, predictor = _model_and_predictor(seed=5)
    clean = toy_latents(batch=4, seed=5)
    timesteps = torch.tensor([100, 300, 600, 900])
    cond = torch.randn((4, 4, 5), generator=torch.Generator().manual_seed(6))
    scorer = DenoisingLossScorer(clean, timesteps, cond, prediction_type="epsilon", autocast_dtype=NO_AUTOCAST)
    baseline = scorer.calibrate(predictor)
    assert math.isfinite(baseline) and baseline > 0
    perfect = FixedPredictor(scorer.clean_target)
    assert scorer.score(perfect) == pytest.approx(1.0, abs=1e-5)
    assert scorer.score(predictor) == pytest.approx(0.0, abs=1e-5)


# -----------------------------------------------------------------------------
# applied_edit safety
# -----------------------------------------------------------------------------


def test_applied_edit_restores_bit_exactly_on_exception():
    model, _ = _model_and_predictor(seed=7)
    targets = cross_attention_targets(model)
    before = {t.name: t.weight.clone() for t in targets}
    pointers = {t.name: t.weight.data_ptr() for t in targets}
    edits = [
        LayerEdit(name=t.name, weight=t.weight, alpha=1.0, directions=torch.ones(t.weight.shape[-1]))
        for t in targets
    ]
    with pytest.raises(RuntimeError):
        with applied_edit(model, edits):
            raise RuntimeError("trial crashed")
    for target in targets:
        assert target.weight.data_ptr() == pointers[target.name]
        assert torch.equal(target.weight, before[target.name])


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------


def test_objective_returns_finite_recovery_and_drift():
    model, predictor = _model_and_predictor(seed=11)
    neutral = _neutral_cache(predictor, seed=11)
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        autocast_dtype=NO_AUTOCAST,
    )
    study = create_study()
    study.optimize(objective, n_trials=1, n_jobs=1)
    recovery, drift = study.trials[0].values
    assert math.isfinite(recovery) and math.isfinite(drift)
    assert 0.0 <= recovery <= 1.0
    assert drift >= 0.0


def test_zero_alpha_edit_is_exactly_identity():
    model, predictor = _model_and_predictor(seed=13)
    neutral = _neutral_cache(predictor, seed=13)
    config = TrialConfig(alpha_max_range=(0.0, 0.0), alpha_min_range=(0.0, 0.0))
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        config=config, autocast_dtype=NO_AUTOCAST,
    )
    study = create_study(config)
    study.optimize(objective, n_trials=1, n_jobs=1)
    trial = study.trials[0]
    assert trial.values[1] == 0.0  # identity edit -> zero drift
    assert all(alpha == 0.0 for alpha in trial.user_attrs["alphas"].values())


def test_pareto_front_trials_record_user_attrs():
    model, predictor = _model_and_predictor(seed=17)
    neutral = _neutral_cache(predictor, seed=17)
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        autocast_dtype=NO_AUTOCAST,
    )
    study = create_study()
    study.optimize(objective, n_trials=3, n_jobs=1)
    for trial in study.trials:
        assert set(trial.user_attrs["per_bin"]) == {"low", "mid", "high"}
        assert isinstance(trial.user_attrs["alphas"], dict) and trial.user_attrs["alphas"]
        assert "relative_drift" in trial.user_attrs
        assert trial.user_attrs["projection_mode"] in ("orthogonal", "covariance_regularized")
        assert "edit_spec" in trial.user_attrs


def test_full_strength_edit_produces_positive_drift():
    # Reproduce the edit the objective builds at alpha_max = alpha_min = 1.
    # (Asserting through the study here would be flaky: the sampled range can be
    # empty, start == end, which is a valid zero-layer search point.)
    model, predictor = _model_and_predictor(seed=19)
    neutral = _neutral_cache(predictor, seed=19)
    provider = _identity_provider(model)
    edits = [
        LayerEdit(
            name=target.name,
            weight=target.weight,
            alpha=1.0,
            directions=provider.get(target.name).directions,
        )
        for target in cross_attention_targets(model)
    ]
    with applied_edit(model, edits):
        drift = compute_epsilon_drift(None, predictor, cache=neutral, autocast_dtype=NO_AUTOCAST)
    assert drift > 0.0


def test_covariance_mode_dropped_without_calibration_data():
    model, predictor = _model_and_predictor(seed=23)
    neutral = _neutral_cache(predictor, seed=23)
    config = TrialConfig(projection_modes=("covariance_regularized",))
    with pytest.raises(ValueError, match="feasible projection modes"):
        build_objective(
            model, predictor, _identity_provider(model), AlphaScorer(), neutral,
            config=config, autocast_dtype=NO_AUTOCAST,
        )


def test_covariance_mode_suggested_when_provider_supplies_covariance():
    model, predictor = _model_and_predictor(seed=29)
    neutral = _neutral_cache(predictor, seed=29)
    entries = {}
    for target in cross_attention_targets(model):
        dim = target.weight.shape[-1]
        entries[target.name] = SubspaceEntry(directions=torch.ones(dim, 1), covariance=torch.eye(dim))
    config = TrialConfig(projection_modes=("covariance_regularized",))
    objective = build_objective(
        model, predictor, DictSubspaceProvider(entries), AlphaScorer(), neutral,
        config=config, autocast_dtype=NO_AUTOCAST,
    )
    study = create_study(config)
    study.optimize(objective, n_trials=1, n_jobs=1)
    assert study.trials[0].user_attrs["projection_mode"] == "covariance_regularized"
    assert study.trials[0].user_attrs["lambda_reg"] is not None


def test_mask_layer_mode_runs():
    model, predictor = _model_and_predictor(seed=31)
    neutral = _neutral_cache(predictor, seed=31)
    config = TrialConfig(layer_mode="mask")
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        config=config, autocast_dtype=NO_AUTOCAST,
    )
    study = create_study(config)
    study.optimize(objective, n_trials=1, n_jobs=1)
    assert len(study.trials[0].user_attrs["alphas"]) == len(cross_attention_targets(model))


def test_block_indexing_preserves_depth_semantics():
    model, predictor = _model_and_predictor(seed=33, depth=3)
    neutral = _neutral_cache(predictor, seed=33)
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        autocast_dtype=NO_AUTOCAST,
    )
    study = create_study()
    study.optimize(objective, n_trials=5, n_jobs=1)
    for trial in study.trials:
        # ToyUNet depth=3 has 3 cross-attention blocks (blocks.0, blocks.1, blocks.2)
        assert trial.user_attrs["num_blocks"] == 3
        start, end = trial.user_attrs["block_range"]
        assert 0 <= start <= end <= 3
        # Check that targets within selected blocks have matching alpha
        alphas = trial.user_attrs["alphas"]
        target_proj = trial.user_attrs["target_projection"]
        for b in range(3):
            k_alpha = alphas.get(f"blocks.{b}.to_k", 0.0)
            v_alpha = alphas.get(f"blocks.{b}.to_v", 0.0)
            if not (start <= b < end):
                assert k_alpha == 0.0 and v_alpha == 0.0
            else:
                if target_proj == "both":
                    assert k_alpha == v_alpha
                elif target_proj == "to_k":
                    assert v_alpha == 0.0
                elif target_proj == "to_v":
                    assert k_alpha == 0.0


def test_weights_restored_after_study():
    model, predictor = _model_and_predictor(seed=37)
    neutral = _neutral_cache(predictor, seed=37)
    before = {t.name: t.weight.clone() for t in cross_attention_targets(model)}
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        autocast_dtype=NO_AUTOCAST,
    )
    study = create_study()
    study.optimize(objective, n_trials=3, n_jobs=1)
    for target in cross_attention_targets(model):
        assert torch.equal(target.weight, before[target.name])


def test_reference_match_scorer_end_to_end_produces_finite_values():
    torch.manual_seed(41)
    reference_model = ToyUNet(width=8, context=5, depth=2)
    erased_model = copy.deepcopy(reference_model)
    # Simulate an erasure: perturb every to_v weight slightly.
    for target in cross_attention_targets(erased_model, include=("to_v",)):
        with torch.no_grad():
            target.weight.mul_(0.8)
    reference_predictor = UNetPredictor(reference_model, prediction_type="epsilon")
    erased_predictor = UNetPredictor(erased_model, prediction_type="epsilon")

    latents = toy_latents(batch=5, seed=41)
    cond = torch.randn((5, 4, 5), generator=torch.Generator().manual_seed(42))
    concept_cache = NoiseCache.build(
        reference_predictor, latents, None, cond, seed=43, autocast_dtype=NO_AUTOCAST
    )
    neutral = _neutral_cache(erased_predictor, seed=44)
    scorer = ReferenceMatchScorer(reference_predictor, concept_cache, autocast_dtype=NO_AUTOCAST)
    scorer.calibrate(erased_predictor)

    provider = _identity_provider(erased_model)
    objective = build_objective(
        erased_model, erased_predictor, provider, scorer, neutral,
        concept_cache=concept_cache, autocast_dtype=NO_AUTOCAST,
    )
    study = create_study()
    study.optimize(objective, n_trials=4, n_jobs=1)
    for trial in study.trials:
        recovery, drift = trial.values
        assert math.isfinite(recovery) and math.isfinite(drift)
        assert 0.0 <= recovery <= 1.0 and drift >= 0.0


# -----------------------------------------------------------------------------
# Study configuration
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", ["nsgaii", "tpe"])
def test_sampler_configuration(sampler):
    config = TrialConfig(sampler=sampler)
    study = create_study(config)
    assert study.directions == [
        optuna.study.StudyDirection.MAXIMIZE,
        optuna.study.StudyDirection.MINIMIZE,
    ]


def test_invalid_sampler_rejected():
    with pytest.raises(ValueError, match="sampler"):
        create_study(TrialConfig(sampler="random"))


def test_benign_default_concepts_are_injectable_and_safe():
    # Defaults are benign Imagenette-style prompts; the list is always a parameter.
    assert BENIGN_CONCEPT_PROMPTS and all("photo of" in prompt for prompt in BENIGN_CONCEPT_PROMPTS)
    assert TrialConfig().projection_modes[0] == "orthogonal"


# -----------------------------------------------------------------------------
# Benchmark + generative validator
# -----------------------------------------------------------------------------


def test_benchmark_trial_reports_reasonable_timing():
    model, predictor = _model_and_predictor(seed=47)
    neutral = _neutral_cache(predictor, seed=47)
    objective = build_objective(
        model, predictor, _identity_provider(model), AlphaScorer(), neutral,
        autocast_dtype=NO_AUTOCAST,
    )
    result = benchmark_trial(objective, n_trials=3, warmup=1)
    assert result["trials"] == 3.0
    assert result["per_trial_sec"] > 0
    # Generous threshold: the toy trial must be far under two seconds on CPU.
    assert result["per_trial_sec"] < 2.0, result


class _StubClassifier:
    def score_image(self, image: torch.Tensor) -> float:
        return 0.9


def test_generative_validator_uses_injected_sampler_and_classifier():
    def sampler(predictor, concept, num_samples, seed):
        return torch.zeros((num_samples, 3, 4, 4))

    validator = GenerativeValidator(sampler=sampler, classifier=_StubClassifier())
    result = validator.validate(model=None, concept="tench", num_samples=3, seed=0)
    assert result.concept == "tench"
    assert result.recovery_score == pytest.approx(0.9)
    assert result.cost["sample_count"] == 3


def test_generative_validator_requires_sampler_or_scheduler():
    validator = GenerativeValidator(classifier=_StubClassifier())
    with pytest.raises(RuntimeError, match="injected sampler or scheduler"):
        validator.validate(model=None, concept="tench", num_samples=1, seed=0)


def test_generative_validator_requires_classifier():
    def sampler(predictor, concept, num_samples, seed):
        return torch.zeros((num_samples, 3, 4, 4))

    validator = GenerativeValidator(sampler=sampler, classifier=None)
    with pytest.raises(RuntimeError, match="RecallClassifier"):
        validator.validate(model=None, concept="tench", num_samples=1, seed=0)