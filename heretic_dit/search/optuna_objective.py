"""Multi-objective Optuna search over projection edits.

The search trades two objectives:

* **maximize** concept recovery - did the cheap edit undo part of the erasure?
* **minimize** neutral prediction drift - is collateral damage acceptable?

Everything expensive is hoisted out of the trial loop: the neutral and concept
:class:`~heretic_dit.metrics.drift.NoiseCache` objects are built once, the
concept subspaces are resolved once, and each trial only clones the touched
weight tensors, applies the edit, runs the cached forwards, and restores.

Concepts are benign proxies only (Imagenette classes, artist styles, benchmark
celebrities); the concept list is always an injectable parameter with benign
defaults.  Nothing here hardcodes concepts.

Protocol notes
--------------
* ``SubspaceProvider`` is consumed through a :func:`resolve_subspace` adapter
  that accepts either the task-spec ``get(layer_name)`` form or the shared
  ``heretic_dit.interfaces`` form (``get_subspace`` / ``get_covariance``), so
  this module works whether or not :mod:`heretic_dit.core.subspace` is finished.
* ``ConceptScorer`` follows the task-spec ``score(edited_predictor)`` contract.
  The shared ``interfaces.ConceptScorer`` scores a representation with a
  concept label instead; that is a different (embedding-level) stage, so the two
  are deliberately not the same object.

Optuna does not support pruning in multi-objective studies, so no pruners are
created here.  The objective mutates live weights, so studies must run with
``n_jobs=1`` (an internal lock is held during each edit to make that explicit).
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import optuna
import torch
from torch import Tensor

from heretic_dit.interfaces import ProxyEvaluation, RecoveryResult
from heretic_dit.metrics.drift import (
    NoiseCache,
    NoisePredictor,
    add_noise,
    compute_epsilon_drift,
    linear_beta_schedule,
)
from heretic_dit.search.editing import (
    InterpolationKernel,
    LayerEdit,
    ProjectionTarget,
    applied_edit,
    cross_attention_targets,
    resolve_alphas,
)

__all__ = [
    "SubspaceProvider",
    "SubspaceEntry",
    "DictSubspaceProvider",
    "ConceptScorer",
    "ReferenceMatchScorer",
    "DenoisingLossScorer",
    "RecallClassifier",
    "GenerativeValidator",
    "TrialConfig",
    "build_objective",
    "create_study",
    "benchmark_trial",
    "resolve_subspace",
    "BENIGN_CONCEPT_PROMPTS",
]

LOGGER = logging.getLogger(__name__)

# Benign proxy concepts by default, mirroring the Imagenette class list used for
# erasure auditing.  Always overridable; never hardcode concepts downstream.
BENIGN_CONCEPT_PROMPTS: Tuple[str, ...] = (
    "a photo of a tench",
    "a photo of an English springer",
    "a photo of a cassette player",
    "a photo of a chain saw",
    "a photo of a church",
    "a photo of a French horn",
    "a photo of a garbage truck",
    "a photo of a gas pump",
    "a photo of a golf ball",
    "a photo of a parachute",
)


# -----------------------------------------------------------------------------
# Subspace provider protocol + adapter
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SubspaceEntry:
    """Projector-ready inputs for one layer.

    Attributes:
        directions: ``(d, k)`` direction columns (or a single ``(d,)`` vector)
            consumed directly by :func:`heretic_dit.core.projector.project_weights`.
        neutral: Optional ``(samples, d)`` calibration activations for covariance mode.
        covariance: Optional ``(d, d)`` precomputed covariance for covariance mode.
    """

    directions: Tensor
    neutral: Optional[Tensor] = None
    covariance: Optional[Tensor] = None

    @property
    def supports_covariance(self) -> bool:
        return self.neutral is not None or self.covariance is not None


@runtime_checkable
class SubspaceProvider(Protocol):
    """Supplies the projector's concept directions (and optional covariance).

    The task-spec form is ``get(layer_name)`` returning a :class:`SubspaceEntry`
    (or a raw directions tensor / ``(directions, neutral)`` tuple).  The shared
    ``heretic_dit.interfaces.SubspaceProvider`` form exposes
    ``get_subspace(concept, layer_name, dim)`` and ``get_covariance``; both are
    accepted by :func:`resolve_subspace`.
    """

    def get(self, layer_name: str) -> Any:
        """Return the projection inputs for ``layer_name``."""
        ...


def _normalize_entry(value: Any, layer_name: str) -> SubspaceEntry:
    if isinstance(value, SubspaceEntry):
        return value
    if isinstance(value, Tensor):
        return SubspaceEntry(directions=value)
    if isinstance(value, Mapping):
        return SubspaceEntry(
            directions=value["directions"],
            neutral=value.get("neutral"),
            covariance=value.get("covariance"),
        )
    if isinstance(value, (tuple, list)) and value:
        return SubspaceEntry(
            directions=value[0],
            neutral=value[1] if len(value) > 1 else None,
            covariance=value[2] if len(value) > 2 else None,
        )
    raise TypeError(f"Unsupported subspace value for {layer_name!r}: {type(value).__name__}.")


def resolve_subspace(
    provider: Any, layer_name: str, dim: int, *, concept: str = "", need_covariance: bool = False
) -> SubspaceEntry:
    """Adapt any supported provider shape to a :class:`SubspaceEntry`.

    Resolution order: explicit ``resolve(...)`` hook, then the task-spec
    ``get(layer_name)``, then the shared ``get_subspace`` / ``get_covariance``
    pair.  ``need_covariance`` requests the covariance branch when available.
    """
    if hasattr(provider, "resolve"):
        return _normalize_entry(provider.resolve(layer_name, dim), layer_name)
    if hasattr(provider, "get"):
        return _normalize_entry(provider.get(layer_name), layer_name)
    if hasattr(provider, "get_subspace"):
        directions = provider.get_subspace(concept, layer_name, dim)
        covariance = None
        if need_covariance and hasattr(provider, "get_covariance"):
            covariance = provider.get_covariance(layer_name, dim)
        return SubspaceEntry(directions=directions, covariance=covariance)
    raise TypeError("provider must expose get(), resolve(), or get_subspace().")


class DictSubspaceProvider:
    """Trivial provider backed by a ``{layer_name: entry}`` mapping.

    Used by tests and by callers that precompute subspaces once.  Missing layers
    are an explicit error rather than a silently empty edit.
    """

    def __init__(self, entries: Mapping[str, Any]):
        self._entries = dict(entries)

    def get(self, layer_name: str) -> Any:
        if layer_name not in self._entries:
            raise KeyError(f"No subspace registered for layer {layer_name!r}.")
        return self._entries[layer_name]

    def __contains__(self, layer_name: object) -> bool:
        return layer_name in self._entries


# -----------------------------------------------------------------------------
# Concept scorers
# -----------------------------------------------------------------------------


@runtime_checkable
class ConceptScorer(Protocol):
    """Fast proxy for concept recovery on cached inputs (higher = more recovered)."""

    def score(self, edited_predictor: NoisePredictor, concept_cache: Optional[NoiseCache] = None) -> float:
        """Return a score in ``[0, 1]`` (0 = no recovery, 1 = full recovery)."""
        ...


def _mean_squared_error(prediction: Tensor, target: Tensor) -> float:
    return float((prediction.float() - target.float()).pow(2).mean())


def _normalized_recovery(mse: float, baseline: float, eps: float = 1e-12) -> float:
    """Recovery in ``[0, 1]`` from an MSE and the erased model's own baseline MSE."""
    if baseline <= eps:
        return 1.0 if mse <= eps else 0.0
    return float(min(1.0, max(0.0, (baseline - mse) / baseline)))


@dataclass
class ReferenceMatchScorer:
    """Proxy recovery = match to the ORIGINAL unerased model's predictions.

    Built on a concept :class:`NoiseCache` whose ``base_pred`` holds the
    *reference* (unerased) model's predictions.  Recovery is

    ``1 - mse(edited, reference) / mse(erased, reference)``

    clipped to ``[0, 1]``: 0 means the edit changed nothing relative to the
    erased model, 1 means the edited model matches the reference exactly.
    The denominator is measured once by :meth:`calibrate` from the unedited
    checkpoint, before the study starts.

    Args:
        reference_predictor: The ORIGINAL unerased model (target of recovery).
        concept_cache: Cache built with ``reference_predictor`` as its base.
        erased_reference_mse: Optional precomputed baseline; if omitted,
            :meth:`calibrate` must be called with the erased predictor first.
        autocast_dtype: Forward autocast dtype (``None`` keeps toy models exact).
    """

    reference_predictor: Any
    concept_cache: Optional[NoiseCache] = None
    erased_reference_mse: Optional[float] = None
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16

    def calibrate(self, erased_predictor: NoisePredictor) -> float:
        """Measure the erased model's MSE to the reference; call once, up front."""
        if self.concept_cache is None:
            raise ValueError("concept_cache is required to calibrate.")
        with torch.inference_mode():
            prediction = erased_predictor.predict_noise(
                self.concept_cache.x_t, self.concept_cache.t, self.concept_cache.cond
            )
        self.erased_reference_mse = _mean_squared_error(prediction, self.concept_cache.base_pred)
        return self.erased_reference_mse

    def score(self, edited_predictor: NoisePredictor, concept_cache: Optional[NoiseCache] = None) -> float:
        cache = concept_cache or self.concept_cache
        if cache is None:
            raise ValueError("concept_cache is required to score.")
        if self.erased_reference_mse is None:
            raise ValueError("Call calibrate(erased_predictor) before scoring.")
        with torch.inference_mode():
            prediction = edited_predictor.predict_noise(cache.x_t, cache.t, cache.cond)
        mse = _mean_squared_error(prediction, cache.base_pred)
        return _normalized_recovery(mse, self.erased_reference_mse)


@dataclass
class DenoisingLossScorer:
    """Proxy recovery = denoising error against the analytic clean target.

    Uses real concept images noised to ``x_t``.  Because the clean latent and the
    sampled ``x_t`` are both known, the ground-truth epsilon / velocity is exact:

    * epsilon: ``(x_t - sqrt(ab) x_0) / sqrt(1 - ab)``
    * velocity: ``sqrt(ab) eps - sqrt(1 - ab) x_0``

    Recovers fewer assumptions than :class:`ReferenceMatchScorer` (no reference
    model needed) but loses to it when the reference is available, because the
    clean target is deterministic rather than distribution-matched.

    Args:
        clean_latents: ``(B, ...)`` encoded concept images.
        timesteps: Per-sample timesteps used to add noise.
        cond: Concept conditioning.
        prediction_type: ``"epsilon"`` or ``"velocity"``.
        erased_baseline: Optional precomputed baseline MSE (see :meth:`calibrate`).
    """

    clean_latents: Tensor
    timesteps: Tensor
    cond: Any
    prediction_type: str = "epsilon"
    seed: int = 0
    erased_baseline: Optional[float] = None
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16

    def __post_init__(self) -> None:
        if self.prediction_type not in ("epsilon", "velocity"):
            raise ValueError("prediction_type must be 'epsilon' or 'velocity'.")
        self._build()

    def _build(self) -> None:
        timesteps = self.timesteps.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        bar = linear_beta_schedule()
        ab = bar[timesteps].to(device=self.clean_latents.device, dtype=torch.float64)
        view = ab.reshape(ab.shape[0], *([1] * (self.clean_latents.ndim - 1)))
        generator = torch.Generator(device="cpu").manual_seed(int(self.seed))
        noise = torch.randn(
            self.clean_latents.shape, generator=generator,
            dtype=torch.float32, device="cpu",
        ).to(device=self.clean_latents.device, dtype=self.clean_latents.dtype)
        self.x_t = add_noise(self.clean_latents, timesteps, noise, bar)
        self.t = timesteps.to(device=self.clean_latents.device)
        sqrt_ab = view.sqrt()
        sqrt_one_minus = (1.0 - view).clamp_min(0).sqrt()
        epsilon = (self.x_t.to(torch.float64) - sqrt_ab * self.clean_latents.to(torch.float64)) / sqrt_one_minus.clamp_min(1e-8)
        if self.prediction_type == "epsilon":
            self.clean_target = epsilon.to(self.clean_latents.dtype)
        else:
            velocity = sqrt_ab * epsilon - sqrt_one_minus * self.clean_latents.to(torch.float64)
            self.clean_target = velocity.to(self.clean_latents.dtype)

    def calibrate(self, erased_predictor: NoisePredictor) -> float:
        """Measure the erased model's denoising loss; call once, up front."""
        with torch.inference_mode():
            prediction = erased_predictor.predict_noise(self.x_t, self.t, self.cond)
        self.erased_baseline = _mean_squared_error(prediction, self.clean_target)
        return self.erased_baseline

    def score(self, edited_predictor: NoisePredictor, concept_cache: Optional[NoiseCache] = None) -> float:
        del concept_cache  # denoising loss derives its own target
        if self.erased_baseline is None:
            raise ValueError("Call calibrate(erased_predictor) before scoring.")
        with torch.inference_mode():
            prediction = edited_predictor.predict_noise(self.x_t, self.t, self.cond)
        mse = _mean_squared_error(prediction, self.clean_target)
        return _normalized_recovery(mse, self.erased_baseline)


# -----------------------------------------------------------------------------
# Generative validation (final Pareto front only - never in the trial loop)
# -----------------------------------------------------------------------------


@runtime_checkable
class RecallClassifier(Protocol):
    """Pluggable image classifier; higher = the concept is more present."""

    def score_image(self, image: Tensor) -> float:
        """Return a confidence in ``[0, 1]`` that ``image`` depicts the concept."""
        ...


SamplerCallable = Callable[[NoisePredictor, str, int, int], Tensor]


@dataclass
class GenerativeValidator:
    """Confirms the fast proxy with real, few-step generation.

    This is intentionally *not* part of the Optuna loop: it runs only on the
    final Pareto front, on a handful of images per concept.  Both the sampler and
    the classifier are injectable so tests never need a real pipeline.

    Args:
        reference_model: Original unerased predictor (for reference-based metrics).
        classifier: Injected :class:`RecallClassifier`.
        sampler: ``(predictor, concept, num_samples, seed) -> images`` callable.
            When ``None``, :meth:`validate` requires ``scheduler`` and uses a
            minimal iterative denoiser.
        scheduler: Optional diffusers-style scheduler providing
            ``timesteps`` / ``step`` and ``add_noise`` via the module helper.
        num_inference_steps: Steps for the few-step sampler.
        concept_prompt_fn: Maps a concept label to a conditioning payload.
    """

    reference_model: Any = None
    classifier: Optional[RecallClassifier] = None
    sampler: Optional[SamplerCallable] = None
    scheduler: Any = None
    num_inference_steps: int = 8
    concept_prompt_fn: Optional[Callable[[str], Any]] = None

    def validate(
        self,
        model: Any,
        concept: str,
        num_samples: int = 4,
        seed: int = 0,
        *,
        predictor: Optional[NoisePredictor] = None,
        edit_context: Optional[Callable[[], Any]] = None,
    ) -> RecoveryResult:
        """Generate and score ``num_samples`` images; return a :class:`RecoveryResult`.

        ``edit_context`` is an optional zero-arg context-manager factory that
        applies (and restores) the candidate edit around generation, so the
        validator can be pointed at a real model + edit pair.
        """
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        active_predictor = predictor if predictor is not None else model
        context = edit_context() if edit_context is not None else _null_context()
        with context:
            images = self._generate(active_predictor, concept, num_samples, seed)
        scores = self._classify(images, concept)
        recovery = float(sum(scores) / len(scores)) if scores else 0.0
        return RecoveryResult(
            concept=concept,
            method="heretic-dit",
            recovery_score=recovery,
            drift_score=float("nan"),
            metrics={"num_samples": float(num_samples), "min_score": min(scores, default=0.0),
                     "max_score": max(scores, default=0.0)},
            cost={"sample_count": num_samples},
        )

    def _generate(self, predictor: Any, concept: str, num_samples: int, seed: int) -> Tensor:
        if self.sampler is not None:
            return self.sampler(predictor, concept, num_samples, seed)
        if self.scheduler is None:
            raise RuntimeError(
                "GenerativeValidator needs an injected sampler or scheduler; "
                "generation is deliberately not part of the search loop."
            )
        return self._denoise(predictor, concept, num_samples, seed)

    def _conditioning(self, concept: str) -> Any:
        if self.concept_prompt_fn is None:
            raise RuntimeError("concept_prompt_fn is required for scheduler-based generation.")
        return self.concept_prompt_fn(concept)

    def _denoise(self, predictor: NoisePredictor, concept: str, num_samples: int, seed: int) -> Tensor:
        scheduler = self.scheduler
        shape = getattr(scheduler, "latent_shape", None)
        if shape is None:
            raise RuntimeError("scheduler must expose a latent_shape for generation.")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latents = torch.randn((num_samples, *shape), generator=generator, dtype=torch.float32)
        conditioning = self._conditioning(concept)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            timesteps = t.expand(num_samples) if hasattr(t, "expand") else torch.full((num_samples,), int(t))
            with torch.inference_mode():
                prediction = predictor.predict_noise(latents, timesteps, conditioning)
            latents = scheduler.step(prediction, t, latents).prev_sample
        return latents

    def _classify(self, images: Tensor, concept: str) -> List[float]:
        if self.classifier is None:
            raise RuntimeError("A RecallClassifier must be injected to score generated images.")
        scores: List[float] = []
        for index in range(images.shape[0]):
            scores.append(float(self.classifier.score_image(images[index])))
        return scores


@contextmanager
def _null_context() -> Iterator[None]:
    yield


# -----------------------------------------------------------------------------
# Search space
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialConfig:
    """Static bounds of the search space.

    Attributes:
        layer_mode: ``"mask"`` (one boolean per block) or ``"range"`` (start/end).
            ``"range"`` is the default to keep the dimension low.
        alpha_max_range / alpha_min_range / peak_position_range / falloff_range:
            Bounds for the Heretic-style per-layer alpha kernel.
        lambda_reg_range: Log-uniform bounds for covariance regularization.
        target_projection: Default categorical value for which projections to hit.
        projection_modes: Categorical modes offered; covariance is dropped
            automatically when the provider cannot supply calibration data.
        sampler: ``"nsgaii"`` (default) or ``"tpe"`` (multi-objective TPE).
        side: Projection side handed to the projector.
    """

    layer_mode: str = "range"
    alpha_max_range: Tuple[float, float] = (0.0, 2.0)
    alpha_min_range: Tuple[float, float] = (0.0, 0.5)
    peak_position_range: Tuple[float, float] = (0.0, 1.0)
    falloff_range: Tuple[float, float] = (0.05, 1.0)
    lambda_reg_range: Tuple[float, float] = (1e-6, 1e-1)
    target_projection: str = "both"
    projection_modes: Tuple[str, ...] = ("orthogonal", "covariance_regularized")
    sampler: str = "nsgaii"
    side: str = "input"


def build_sampler(config: TrialConfig) -> optuna.samplers.BaseSampler:
    """Return the configured multi-objective sampler (no pruners are created)."""
    if config.sampler == "nsgaii":
        return optuna.samplers.NSGAIISampler()
    if config.sampler == "tpe":
        return optuna.samplers.TPESampler()
    raise ValueError("config.sampler must be 'nsgaii' or 'tpe'.")


def _suggest_alpha(
    trial: optuna.Trial, config: TrialConfig, num_layers: int
) -> Tuple[object, Optional[List[bool]], Optional[int], Optional[int]]:
    """Draw the layer selection and alpha-kernel parameters for one trial."""
    mask = None
    start = end = None
    if config.layer_mode == "mask":
        mask = [trial.suggest_float(f"mask_{i}", 0.0, 1.0, step=1.0) >= 0.5 for i in range(num_layers)]
    elif config.layer_mode == "range":
        start = trial.suggest_int("start", 0, max(num_layers - 1, 0))
        end = trial.suggest_int("end", start, num_layers)
    else:
        raise ValueError("config.layer_mode must be 'mask' or 'range'.")
    alpha_max = trial.suggest_float("alpha_max", *config.alpha_max_range)
    # Derive alpha_min's upper bound from the sampled peak so the kernel is
    # always valid (alpha_min <= alpha_max) without a post-hoc clamp.
    upper_min = min(config.alpha_min_range[1], alpha_max)
    lower_min = min(config.alpha_min_range[0], upper_min)
    kernel = InterpolationKernel(
        alpha_max=alpha_max,
        alpha_min=trial.suggest_float("alpha_min", lower_min, upper_min),
        peak_position=trial.suggest_float("peak_position", *config.peak_position_range),
        falloff_distance=trial.suggest_float("falloff_distance", *config.falloff_range),
    )
    return kernel, mask, start, end


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------


def _layer_dim(weight: Tensor, side: str) -> int:
    return int(weight.shape[-1] if side == "input" else weight.shape[-2])


def build_objective(
    model: Any,
    predictor: NoisePredictor,
    subspace_provider: Any,
    concept_scorer: ConceptScorer,
    neutral_cache: NoiseCache,
    concept_cache: Optional[NoiseCache] = None,
    *,
    targets: Optional[Sequence[ProjectionTarget]] = None,
    config: TrialConfig = TrialConfig(),
    regularization: float = 1e-4,
    compute_dtype: torch.dtype = torch.float64,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    predictor_factory: Optional[Callable[[Any], NoisePredictor]] = None,
    compute_breakdown: bool = True,
) -> Callable[[optuna.Trial], Tuple[float, float]]:
    """Build the Optuna objective ``(recovery, drift)`` for a multi-objective study.

    Everything reusable is resolved here, once: the editable targets, their
    subspaces, and the set of feasible projection modes.  Each trial then only

    1. draws search-space parameters,
    2. clones + edits the touched tensors inside :func:`applied_edit`,
    3. runs the cached forwards for recovery and drift,
    4. restores the weights (guaranteed by the context manager's ``finally``).

    Args:
        model: The live base model whose weights are edited in place.
        predictor: Adapter bound to ``model`` (edits are visible immediately).
        subspace_provider: Provider of concept directions / covariance.
        concept_scorer: Fast recovery proxy (:class:`ConceptScorer`).
        neutral_cache: Cache of neutral inputs + base predictions (drift).
        concept_cache: Cache of concept inputs + reference predictions (recovery).
        targets: Optional pre-discovered targets; defaults to
            :func:`cross_attention_targets`.
        config: Search-space bounds.
        regularization: Ridge passed to the projector for covariance mode when a
            layer entry has no precomputed covariance.
        compute_dtype: Projector arithmetic dtype.
        autocast_dtype: Forward autocast dtype.
        predictor_factory: Optional ``model -> NoisePredictor`` override for the
            edited predictor (defaults to ``predictor``).
        compute_breakdown: Also store per-bin drift on the trial.

    Returns:
        A callable ``objective(trial) -> (recovery, drift)``.

    Note:
        The objective mutates shared weights, so the study must use ``n_jobs=1``.
        A lock is held around each edit to make concurrent misuse fail loudly
        rather than corrupt silently.
    """
    editable = list(targets) if targets is not None else cross_attention_targets(model)
    if not editable:
        raise ValueError("No editable cross-attention targets were found.")
    
    # Identify unique cross-attention blocks (depth order preserved)
    unique_blocks: List[str] = list(
        dict.fromkeys(
            target.name.rsplit(".", 1)[0] if "." in target.name else target.name
            for target in editable
        )
    )
    num_blocks = len(unique_blocks)
    block_to_idx = {name: i for i, name in enumerate(unique_blocks)}
    lock = threading.Lock()

    # Resolve subspaces once; no provider calls inside the trial loop.
    entries: Dict[str, SubspaceEntry] = {}
    for target in editable:
        dim = _layer_dim(target.weight, config.side)
        entries[target.name] = resolve_subspace(
            subspace_provider, target.name, dim, need_covariance=True
        )
    covariance_ready = all(entry.supports_covariance for entry in entries.values())
    modes = tuple(
        mode for mode in config.projection_modes
        if mode != "covariance_regularized" or covariance_ready
    )
    if not modes:
        raise ValueError("No feasible projection modes; provide calibration data for covariance.")

    by_token: Dict[str, List[ProjectionTarget]] = {"to_k": [], "to_v": []}
    for target in editable:
        leaf = target.name.rsplit(".", 1)[-1]
        for token in by_token:
            if token in target.name or token == leaf:
                by_token[token].append(target)
    if config.target_projection != "both" and not by_token.get(config.target_projection):
        raise ValueError(f"No targets matched target_projection={config.target_projection!r}.")

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        kernel, mask, start, end = _suggest_alpha(trial, config, num_blocks)
        block_alphas = resolve_alphas(
            num_blocks, kernel, layer_mode=config.layer_mode, mask=mask, start=start, end=end
        )
        mode = trial.suggest_categorical("projection_mode", list(modes))
        lambda_reg = (
            trial.suggest_float("lambda_reg", *config.lambda_reg_range, log=True)
            if mode == "covariance_regularized"
            else None
        )
        target_projection = trial.suggest_categorical(
            "target_projection", ["to_k", "to_v", "both"]
        )
        tokens = ["to_k", "to_v"] if target_projection == "both" else [target_projection]

        edits: List[LayerEdit] = []
        resolved_alphas: Dict[str, float] = {}
        for target in editable:
            block_name = target.name.rsplit(".", 1)[0] if "." in target.name else target.name
            b_idx = block_to_idx[block_name]
            block_alpha = float(block_alphas[b_idx])
            token = "to_k" if ("to_k" in target.name or target.name.rsplit(".", 1)[-1] == "to_k") else "to_v"

            if token in tokens:
                alpha = block_alpha
            else:
                alpha = 0.0
            resolved_alphas[target.name] = alpha

            if alpha == 0.0:
                continue
            entry = entries[target.name]
            edits.append(
                LayerEdit(
                    name=target.name,
                    weight=target.weight,
                    alpha=alpha,
                    directions=entry.directions,
                    mode=mode,
                    side=config.side,
                    neutral=entry.neutral,
                    covariance=entry.covariance,
                    regularization=lambda_reg if lambda_reg is not None else regularization,
                )
            )

        with lock:
            with applied_edit(model, edits):
                edited_predictor = predictor_factory(model) if predictor_factory else predictor
                recovery = float(concept_scorer.score(edited_predictor, concept_cache))
                breakdown = compute_epsilon_drift(
                    None,
                    edited_predictor,
                    cache=neutral_cache,
                    return_breakdown=compute_breakdown,
                    autocast_dtype=autocast_dtype,
                )
        if compute_breakdown:
            drift = float(breakdown["drift"])
            trial.set_user_attr("per_bin", dict(breakdown["per_bin"]))
            trial.set_user_attr("relative_drift", float(breakdown["relative_drift"]))
        else:
            drift = float(breakdown)

        trial.set_user_attr("recovery", recovery)
        trial.set_user_attr("drift", drift)
        trial.set_user_attr("raw_mse", drift)
        trial.set_user_attr("alphas", resolved_alphas)
        trial.set_user_attr("block_range", (start, end) if start is not None else None)
        trial.set_user_attr("num_blocks", num_blocks)
        trial.set_user_attr("selected_blocks", [b for b, a in zip(unique_blocks, block_alphas) if a > 0])
        trial.set_user_attr("selected_layers", [edit.name for edit in edits])
        trial.set_user_attr("projection_mode", mode)
        trial.set_user_attr("lambda_reg", lambda_reg)
        trial.set_user_attr("target_projection", target_projection)
        trial.set_user_attr("edit_spec", ProxyEvaluation(
            proxy_recovery=recovery, proxy_drift=drift, trial_id=trial.number
        ).__dict__)
        return recovery, drift

    return objective


def create_study(
    config: TrialConfig = TrialConfig(), *, study_name: Optional[str] = None
) -> optuna.Study:
    """Create the two-objective study (maximize recovery, minimize drift)."""
    return optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=build_sampler(config),
        study_name=study_name,
    )


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------


def benchmark_trial(
    objective: Callable[[optuna.Trial], Tuple[float, float]],
    *,
    n_trials: int = 1,
    warmup: int = 0,
    config: TrialConfig = TrialConfig(),
) -> Dict[str, float]:
    """Time the objective over ``n_trials`` trials and log the result.

    Returns a dict with ``trials``, ``total_sec``, ``per_trial_sec`` and
    ``median_trial_sec``.  Warm-up trials are run and discarded so first-call
    kernel/allocator cost does not pollute the measurement.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    study = create_study(config)
    for _ in range(max(warmup, 0)):
        study.optimize(objective, n_trials=1, n_jobs=1)
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    elapsed = time.perf_counter() - start
    durations = [trial.duration.total_seconds() for trial in study.trials[-n_trials:]]
    median = float(torch.tensor(durations, dtype=torch.float64).median()) if durations else elapsed
    result = {
        "trials": float(n_trials),
        "total_sec": elapsed,
        "per_trial_sec": elapsed / n_trials,
        "median_trial_sec": median,
    }
    LOGGER.info(
        "benchmark_trial: %d trials in %.3fs (%.4fs/trial, median %.4fs)",
        n_trials, elapsed, result["per_trial_sec"], median,
    )
    return result