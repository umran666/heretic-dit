"""Prediction-drift metrics for auditing abliteration / erasure robustness.

Terminology
-----------
``base_model``
    The model *before* the projection edit; i.e. the already-erased checkpoint
    being audited (an ESD / MACE / UCE output).
``edited_model``
    ``base_model`` with a cheap, training-free projection edit applied to its
    cross-attention weights.

The central quantity is the mean squared difference between the base and edited
models' noise predictions on a frozen batch of *neutral* (benign control)
inputs.  A concept edit that barely perturbs the neutral distribution is cheap;
one that destroys it is collateral damage.  The same machinery is reused by the
concept-recovery proxy in :mod:`heretic_dit.search.optuna_objective`.

Prediction type
---------------
Diffusion UNets traditionally predict ``epsilon``; flow-matching transformers
(SD3, Flux) predict ``velocity``.  The function is named
:func:`compute_epsilon_drift` for continuity, but it measures drift of whatever
the model predicts.  :func:`compute_prediction_drift` is an alias, and
``prediction_type`` is recorded on the cache / predictor so velocity models are
handled correctly (the *quantity* compared differs; the MSE form does not).

Efficiency
----------
The hot path is a search loop that evaluates one candidate edit per trial:

* :class:`NoiseCache` freezes ``x_t``, ``t``, ``cond`` and the *base* model's
  prediction, computed once up front under :func:`torch.inference_mode`.
* When a cache is supplied, :func:`compute_epsilon_drift` runs only the edited
  model; base predictions are never recomputed.

Forwards run under :func:`torch.inference_mode` with ``requires_grad`` disabled
and no device transfers inside the loop.  Autocast is opt-in through
``autocast_dtype`` (bf16 default on CUDA; pass ``None`` on CPU for bit-exact
toy-model runs).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "PredictionType",
    "NoisePredictor",
    "UNetPredictor",
    "DiTPredictor",
    "NoiseCache",
    "compute_epsilon_drift",
    "compute_prediction_drift",
    "add_noise",
    "linear_beta_schedule",
    "stratified_timesteps",
    "timestep_bins",
]

PredictionType = str  # "epsilon" | "velocity"; kept as str for runtime flexibility.
_VALID_PREDICTION_TYPES = ("epsilon", "velocity")

Conditioning = Any  # Tensor, or a Mapping / sequence of tensors (pooled projections).


# -----------------------------------------------------------------------------
# Predictor adapters
# -----------------------------------------------------------------------------


@runtime_checkable
class NoisePredictor(Protocol):
    """Uniform interface over diffusion backbones.

    Matches ``heretic_dit.interfaces.NoisePredictor`` (method ``predict_noise``)
    and additionally exposes the shorter ``predict`` alias.  ``prediction_type``
    is ``"epsilon"`` or ``"velocity"``.
    """

    prediction_type: PredictionType

    def predict_noise(self, latents: Tensor, timesteps: Tensor, conditioning: Any) -> Tensor:
        """Return the model's prediction for ``latents`` at ``timesteps``."""
        ...

    def predict(self, x_t: Tensor, t: Tensor, cond: Any) -> Tensor:
        """Alias for :meth:`predict_noise` (short spec-compatible name)."""
        ...


def _extract_prediction(output: Any) -> Tensor:
    """Pull the prediction tensor out of a diffusers-style model output."""
    if isinstance(output, Tensor):
        return output
    sample = getattr(output, "sample", None)
    if isinstance(sample, Tensor):
        return sample
    if isinstance(output, Mapping):
        for key in ("sample", "hidden_states", "latents"):
            candidate = output.get(key)
            if isinstance(candidate, Tensor):
                return candidate
        if len(output) == 1:
            only = next(iter(output.values()))
            if isinstance(only, Tensor):
                return only
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Could not extract a prediction tensor from {type(output).__name__}.")


def _validate_prediction_type(prediction_type: PredictionType) -> PredictionType:
    if prediction_type not in _VALID_PREDICTION_TYPES:
        raise ValueError(f"prediction_type must be one of {_VALID_PREDICTION_TYPES}.")
    return prediction_type


@dataclass
class UNetPredictor:
    """Adapter for ``diffusers`` ``UNet2DConditionModel``-style backbones.

    Args:
        model: Callable returning an object with a ``.sample`` tensor.
        prediction_type: ``"epsilon"`` or ``"velocity"``.
        extra_kwargs: Forwarded to the backbone on every call (e.g.
            ``added_cond_kwargs``, ``cross_attention_kwargs``).
    """

    model: Any
    prediction_type: PredictionType = "epsilon"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_prediction_type(self.prediction_type)

    def predict_noise(self, latents: Tensor, timesteps: Tensor, conditioning: Any) -> Tensor:
        output = self.model(latents, timesteps, encoder_hidden_states=conditioning, **self.extra_kwargs)
        return _extract_prediction(output)

    def predict(self, x_t: Tensor, t: Tensor, cond: Any) -> Tensor:
        return self.predict_noise(x_t, t, cond)


@dataclass
class DiTPredictor:
    """Adapter for ``diffusers`` transformer backbones (SD3 / Flux-like).

    Transformer backbones consume ``hidden_states`` / ``timestep`` and,
    optionally, ``encoder_hidden_states`` and pooled text projections.
    ``conditioning`` may be a plain tensor (treated as ``encoder_hidden_states``)
    or a mapping with keys such as ``encoder_hidden_states`` and
    ``pooled_projections``.

    Args:
        model: The transformer module.
        prediction_type: Flow-matching DiTs (SD3, Flux) predict ``"velocity"``.
        pooled_key: Keyword used for pooled text projections.
        extra_kwargs: Forwarded on every call (e.g. ``guidance``).
    """

    model: Any
    prediction_type: PredictionType = "velocity"
    pooled_key: str = "pooled_projections"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_prediction_type(self.prediction_type)

    def _split(self, conditioning: Any) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if isinstance(conditioning, Mapping):
            encoder = conditioning.get("encoder_hidden_states")
            pooled = conditioning.get(self.pooled_key, conditioning.get("pooled_projections"))
            return encoder, pooled
        return conditioning, None

    def predict_noise(self, latents: Tensor, timesteps: Tensor, conditioning: Any) -> Tensor:
        encoder, pooled = self._split(conditioning)
        kwargs: Dict[str, Any] = dict(hidden_states=latents, timestep=timesteps, **self.extra_kwargs)
        if encoder is not None:
            kwargs["encoder_hidden_states"] = encoder
        if pooled is not None:
            kwargs[self.pooled_key] = pooled
        return _extract_prediction(self.model(**kwargs))

    def predict(self, x_t: Tensor, t: Tensor, cond: Any) -> Tensor:
        return self.predict_noise(x_t, t, cond)


# -----------------------------------------------------------------------------
# Noise schedule / deterministic noise helpers
# -----------------------------------------------------------------------------


def linear_beta_schedule(num_train_timesteps: int = 1000) -> Tensor:
    """Return the DDPM linear ``alphas_cumprod`` (CPU float64) of shape ``(T,)``."""
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive.")
    betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0)


def _alphas_cumprod(alphas_cumprod: Optional[Tensor], num_train_timesteps: int) -> Tensor:
    if alphas_cumprod is None:
        return linear_beta_schedule(num_train_timesteps)
    if not isinstance(alphas_cumprod, Tensor) or alphas_cumprod.ndim != 1:
        raise TypeError("alphas_cumprod must be a 1D tensor.")
    return alphas_cumprod.detach().to(device="cpu", dtype=torch.float64)


def _reshape_bar(bar: Tensor, ndim: int) -> Tensor:
    """Reshape per-sample alpha bars to broadcast against a ``(B, ...)`` tensor."""
    return bar.reshape(bar.shape[0], *([1] * (ndim - 1)))


def add_noise(
    clean: Tensor,
    timesteps: Tensor,
    noise: Tensor,
    alphas_cumprod: Optional[Tensor] = None,
    *,
    num_train_timesteps: int = 1000,
) -> Tensor:
    """Return ``sqrt(ab) * clean + sqrt(1 - ab) * noise`` on ``clean``'s device/dtype."""
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must share a shape.")
    bar = _alphas_cumprod(alphas_cumprod, num_train_timesteps)
    index = timesteps.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    ab = _reshape_bar(bar[index], clean.ndim).to(device=clean.device, dtype=clean.dtype)
    return ab.sqrt() * clean + (1.0 - ab).sqrt() * noise


def stratified_timesteps(
    num_samples: int,
    *,
    num_train_timesteps: int = 1000,
    num_bins: int = 3,
    seed: int = 0,
) -> Tensor:
    """Deterministically sample timesteps spread across ``num_bins`` noise bins.

    The range ``[0, num_train_timesteps)`` is split into equal-width bins (low /
    mid / high noise); samples are placed round-robin into the bins at a seeded
    jittered position.  This prevents a candidate edit that only perturbs one
    segment of the trajectory from being scored as harmless.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    width = num_train_timesteps / num_bins
    bin_ids = torch.arange(num_samples) % num_bins
    jitter = torch.rand(num_samples, generator=generator, dtype=torch.float64)
    positions = bin_ids.to(torch.float64) * width + jitter * width
    return positions.floor().clamp_(0, num_train_timesteps - 1).to(torch.long)


def timestep_bins(
    timesteps: Tensor, *, num_train_timesteps: int = 1000, num_bins: int = 3
) -> Tensor:
    """Assign each timestep to an equal-width noise bin (0 = low, ...)."""
    if num_bins < 1:
        raise ValueError("num_bins must be positive.")
    ids = timesteps.detach().to(device="cpu", dtype=torch.long)
    edges = num_train_timesteps / num_bins
    return torch.clamp((ids.to(torch.float64) / edges).floor().to(torch.long), 0, num_bins - 1)


_BIN_LABELS = ("low", "mid", "high")


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def _autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None:
        return nullcontext()
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("autocast_dtype must be torch.bfloat16, torch.float16, or None.")
    return torch.autocast(device_type=device.type, dtype=dtype)


def _take(conditioning: Any, index: slice) -> Any:
    """Slice per-sample conditioning without touching shared/global entries."""
    if conditioning is None or isinstance(conditioning, Tensor):
        return conditioning if conditioning is None else conditioning[index]
    if isinstance(conditioning, Mapping):
        return {key: _take(value, index) for key, value in conditioning.items()}
    if isinstance(conditioning, (list, tuple)):
        return type(conditioning)(_take(value, index) for value in conditioning)
    return conditioning


def _conditioning_batch(conditioning: Any) -> int:
    if isinstance(conditioning, Tensor):
        return conditioning.shape[0]
    if isinstance(conditioning, Mapping):
        values: Sequence[Any] = list(conditioning.values())
    elif isinstance(conditioning, (list, tuple)):
        values = list(conditioning)
    else:
        return -1
    for value in values:
        if isinstance(value, Tensor) and value.ndim > 0:
            return value.shape[0]
    return -1


# -----------------------------------------------------------------------------
# Frozen noise cache
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseCache:
    """Frozen inputs and base predictions reused across every trial.

    Attributes:
        x_t: Noisy latents, shape ``(B, ...)``.
        t: Per-sample timesteps, shape ``(B,)``.
        cond: Conditioning passed to the backbone (tensor or mapping).
        base_pred: Base model prediction on ``(x_t, t, cond)``.
        bins: Per-sample noise-bin id in ``[0, num_bins)``.
        prediction_type: ``"epsilon"`` or ``"velocity"``.
        uncond_cond: Optional unconditional conditioning (for CFG drift).
        base_uncond_pred: Base unconditional prediction, if ``uncond_cond`` given.
    """

    x_t: Tensor
    t: Tensor
    cond: Any
    base_pred: Tensor
    bins: Tensor
    prediction_type: PredictionType = "epsilon"
    uncond_cond: Any = None
    base_uncond_pred: Optional[Tensor] = None

    def __len__(self) -> int:
        return int(self.x_t.shape[0])

    @staticmethod
    def build(
        base_predictor: NoisePredictor,
        latents: Tensor,
        timesteps: Optional[Tensor],
        cond: Any,
        seed: int = 0,
        *,
        uncond_cond: Any = None,
        alphas_cumprod: Optional[Tensor] = None,
        num_train_timesteps: int = 1000,
        num_bins: int = 3,
        autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    ) -> "NoiseCache":
        """Build the cache; this is the one-time up-front cost of a study.

        Args:
            base_predictor: The model *before* the edit.
            latents: Clean latents to which deterministic noise is added.
            timesteps: Optional per-sample timesteps.  ``None`` triggers
                stratified sampling across ``num_bins`` noise bins.
            cond: Conditioning matching ``latents``.
            seed: Pins the noise realisation so every trial sees identical ``x_t``.
            uncond_cond: Optional unconditional conditioning for CFG measurements.
            alphas_cumprod: Optional schedule; defaults to linear beta DDPM.
            num_bins: Number of stratified noise bins.
            autocast_dtype: Autocast dtype for the base forward (bf16 default).
        """
        if not isinstance(latents, Tensor) or latents.ndim < 2:
            raise TypeError("latents must be a tensor of shape (B, ...).")
        batch = int(latents.shape[0])
        if timesteps is None:
            timesteps = stratified_timesteps(
                batch, num_train_timesteps=num_train_timesteps, num_bins=num_bins, seed=seed
            )
        timesteps = timesteps.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if timesteps.shape[0] != batch:
            raise ValueError("timesteps must have one entry per latent sample.")
        conditioning_batch = _conditioning_batch(cond)
        if conditioning_batch not in (-1, batch):
            raise ValueError("conditioning batch does not match latents.")
        schedule = _alphas_cumprod(alphas_cumprod, num_train_timesteps)

        generator = _device_generator(latents.device, seed)
        noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32, device=latents.device)
        x_t = add_noise(latents, timesteps, noise.to(latents.dtype), schedule)

        device_t = timesteps.to(device=latents.device)
        with torch.inference_mode(), _autocast_context(latents.device, autocast_dtype):
            base_pred = base_predictor.predict_noise(x_t, device_t, cond).detach()
            base_uncond_pred = None
            if uncond_cond is not None:
                base_uncond_pred = base_predictor.predict_noise(x_t, device_t, uncond_cond).detach()
        bins = timestep_bins(timesteps, num_train_timesteps=num_train_timesteps, num_bins=num_bins)
        return NoiseCache(
            x_t=x_t,
            t=device_t,
            cond=cond,
            base_pred=base_pred,
            bins=bins,
            prediction_type=getattr(base_predictor, "prediction_type", "epsilon"),
            uncond_cond=uncond_cond,
            base_uncond_pred=base_uncond_pred,
        )


# -----------------------------------------------------------------------------
# Drift
# -----------------------------------------------------------------------------


def _guided(cond_pred: Tensor, uncond_pred: Tensor, cfg_scale: float) -> Tensor:
    return uncond_pred + cfg_scale * (cond_pred - uncond_pred)


def _forward(
    predictor: NoisePredictor,
    x_t: Tensor,
    t: Tensor,
    cond: Any,
    autocast_dtype: Optional[torch.dtype],
    device: torch.device,
) -> Tensor:
    with _autocast_context(device, autocast_dtype):
        return predictor.predict_noise(x_t, t, cond)


def compute_epsilon_drift(
    base_model: Optional[NoisePredictor],
    edited_model: NoisePredictor,
    latents_t: Optional[Tensor] = None,
    timesteps: Optional[Tensor] = None,
    neutral_conditioning: Any = None,
    *,
    cache: Optional[NoiseCache] = None,
    batch_size: int = 8,
    return_breakdown: bool = False,
    cfg_scale: Optional[float] = None,
    uncond_conditioning: Any = None,
    uncond_model: Optional[NoisePredictor] = None,
    prediction_type: Optional[PredictionType] = None,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Union[float, Dict[str, Any]]:
    """Mean squared prediction drift between the base and edited models.

    ``base_model`` is the erased checkpoint being audited; ``edited_model`` is
    that checkpoint with a projection edit applied.  For flow-matching models
    the compared quantity is velocity rather than epsilon; the MSE form is
    unchanged (see module docstring), hence the :func:`compute_prediction_drift`
    alias.

    Args:
        base_model: Base predictor.  May be ``None`` when ``cache`` is supplied,
            because base predictions then come from the cache.
        edited_model: Predictor under test (the live, edited model).
        latents_t: Noisy latents ``x_t``.  Ignored when ``cache`` is given.
        timesteps: Per-sample timesteps.  Ignored when ``cache`` is given.
        neutral_conditioning: Conditioning for the neutral batch.  Ignored when
            ``cache`` is given.
        cache: Pre-built :class:`NoiseCache`.  When present, only the edited
            model runs (the hot path).
        batch_size: Samples per forward pass.
        return_breakdown: Return per-bin and relative drift instead of a scalar.
        cfg_scale: If given, drift is measured on the CFG-guided prediction
            ``uncond + scale * (cond - uncond)``, which amplifies any shift.
            Requires unconditional conditioning.
        uncond_conditioning: Unconditional embedding for CFG.  Taken from the
            cache when omitted.
        uncond_model: Optional separate predictor for the unconditional branch.
            Defaults to ``edited_model`` (and, for the base branch, to the
            cached base unconditional prediction).
        prediction_type: Overrides the value recorded on the cache/predictor.
        autocast_dtype: Autocast dtype for forwards (bf16 default; ``None``
            disables autocast, keeping toy-model runs bit-exact).

    Returns:
        A float MSE, or, when ``return_breakdown`` is true, a dict with keys
        ``drift``, ``relative_drift``, ``per_bin`` (low/mid/high),
        ``num_samples``, ``elements``, ``prediction_type`` and ``cfg_scale``.
    """
    if cache is not None:
        x_t, t, cond, base_pred, bins = cache.x_t, cache.t, cache.cond, cache.base_pred, cache.bins
        if uncond_conditioning is None:
            uncond_conditioning = cache.uncond_cond
        if prediction_type is None:
            prediction_type = cache.prediction_type
        base_uncond_pred = cache.base_uncond_pred
    else:
        if base_model is None:
            raise ValueError("base_model is required when no cache is supplied.")
        if not isinstance(latents_t, Tensor) or timesteps is None:
            raise ValueError("latents_t and timesteps are required when no cache is supplied.")
        x_t = latents_t
        t = timesteps.detach().to(device=x_t.device, dtype=torch.long).reshape(-1)
        cond = neutral_conditioning
        bins = timestep_bins(t)
        base_pred = None
        base_uncond_pred = None
    if prediction_type is None:
        prediction_type = getattr(edited_model, "prediction_type", "epsilon")
    _validate_prediction_type(prediction_type)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if cfg_scale is not None and uncond_conditioning is None:
        raise ValueError("cfg_scale requires uncond_conditioning.")

    total = int(x_t.shape[0])
    device = x_t.device
    use_cfg = cfg_scale is not None
    bounds = [(start, min(start + batch_size, total)) for start in range(0, total, batch_size)]

    squared_error = 0.0
    squared_base = 0.0
    elements = 0
    num_bins = int(bins.max()) + 1 if bins.numel() else 0
    bin_squared = torch.zeros(num_bins, dtype=torch.float64)
    bin_count = torch.zeros(num_bins, dtype=torch.float64)
    ones = torch.ones(total, dtype=torch.float64)

    with torch.inference_mode():
        for start, stop in bounds:
            index = slice(start, stop)
            edited_cond = _forward(edited_model, x_t[index], t[index], _take(cond, index), autocast_dtype, device)
            if use_cfg:
                if base_uncond_pred is None:
                    # Uncached base unconditional branch; still computed once.
                    if base_model is None:
                        raise ValueError("CFG drift requires base unconditional predictions.")
                    base_uncond_pred = _forward(
                        uncond_model or base_model, x_t, t, uncond_conditioning, autocast_dtype, device
                    )
                uncond_pred = _forward(
                    uncond_model or edited_model, x_t[index], t[index],
                    _take(uncond_conditioning, index), autocast_dtype, device,
                )
                reference = _guided(base_pred[index], base_uncond_pred[index], cfg_scale)
                against = _guided(edited_cond, uncond_pred, cfg_scale)
            else:
                if base_pred is None:
                    base_pred = _forward(base_model, x_t, t, cond, autocast_dtype, device)
                reference = base_pred[index]
                against = edited_cond

            error = (against.float() - reference.float()).pow(2)
            squared_error += float(error.sum())
            squared_base += float(reference.float().pow(2).sum())
            elements += reference.numel()
            per_sample = error.reshape(stop - start, -1).sum(dim=1).to(device="cpu", dtype=torch.float64)
            bin_idx = bins[start:stop].to(device="cpu", dtype=torch.long)
            bin_squared.index_add_(0, bin_idx, per_sample)
            bin_count.index_add_(0, bin_idx, ones[start:stop])

    denominator = max(elements, 1)
    drift = squared_error / denominator
    base_magnitude = squared_base / denominator
    relative = drift / base_magnitude if base_magnitude > 0 else float("inf")
    if not return_breakdown:
        return drift

    per_bin: Dict[str, Optional[float]] = {}
    for bin_id in range(num_bins):
        label = _BIN_LABELS[bin_id] if bin_id < len(_BIN_LABELS) else f"bin{bin_id}"
        count = float(bin_count[bin_id])
        per_bin[label] = float(bin_squared[bin_id]) / count if count > 0 else None
    return {
        "drift": drift,
        "relative_drift": relative,
        "per_bin": per_bin,
        "num_samples": total,
        "elements": elements,
        "prediction_type": prediction_type,
        "cfg_scale": cfg_scale,
    }


compute_prediction_drift = compute_epsilon_drift
"""Alias: the metric compares predictions, which may be epsilon or velocity."""