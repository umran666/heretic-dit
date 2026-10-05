"""Paired quality metrics: FID, LPIPS, and CLIP prompt-alignment score.

All comparisons are *paired*: base and edited models are generated from the
same prompts and the same per-image seed derivation (see
``heretic_dit.eval.validator.generate_paired_images``), so quality deltas are
not confounded by sampling noise.

Heavy dependencies are lazy:
    * FID features -- torchvision (InceptionV3); a deterministic
      ``RandomProjectionFeatures`` stand-in is provided for CPU tests.
    * LPIPS -- the ``lpips`` package.
    * CLIP score -- ``open_clip_torch``.
Each raises an ImportError with install instructions on first use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor

from heretic_dit.eval.adapters import GenerationAdapter
from heretic_dit.eval.validator import ValidatorConfig, _generate_with_config

__all__ = [
    "FeatureExtractor",
    "RandomProjectionFeatures",
    "InceptionV3Features",
    "frechet_distance",
    "fid_score",
    "lpips_paired",
    "clip_score",
    "QualityEvaluator",
]


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol for feature extractors used by FID."""

    def extract(self, images: Tensor) -> Tensor:
        """Map an ``(N, C, H, W)`` batch in [0, 1] to ``(N, D)`` features."""
        ...


class RandomProjectionFeatures:
    """Deterministic linear feature extractor for tests and smoke runs.

    Projects flattened images onto a fixed seeded random orthonormal basis;
    FID over these features is a well-defined deterministic function, which
    makes the Frechet-math testable on CPU with no downloads.
    """

    def __init__(self, out_dim: int = 64, seed: int = 0) -> None:
        self._out_dim = out_dim
        self._seed = seed
        self._basis: Optional[Tensor] = None

    def extract(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected (N, C, H, W) images; got {tuple(images.shape)}.")
        flat = images.to(torch.float32).flatten(1)
        in_dim = flat.shape[1]
        if self._basis is None or self._basis.shape[0] != in_dim:
            generator = torch.Generator().manual_seed(self._seed)
            raw = torch.randn(in_dim, self._out_dim, generator=generator)
            self._basis = torch.linalg.qr(raw).Q
        return flat @ self._basis.to(flat.device)


class InceptionV3Features:
    """InceptionV3 pool-3 features (2048-d) via torchvision. Downloads weights."""

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = device
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from torchvision import models as tv_models
            except ImportError as error:  # pragma: no cover - environment-dependent
                raise ImportError(
                    "InceptionV3Features requires torchvision. "
                    "Install it with: pip install torchvision"
                ) from error
            model = tv_models.inception_v3(weights="DEFAULT", transform_input=False)
            model.fc = torch.nn.Identity()
            model.eval()
            self._model = model.to(self._device or "cpu")
        return self._model

    def extract(self, images: Tensor) -> Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        resized = F.interpolate(images.float(), size=(299, 299), mode="bilinear", align_corners=False)
        normalized = (resized - mean) / std
        model = self._ensure_model()
        with torch.no_grad():
            return model(normalized.to(next(model.parameters()).device))


def frechet_distance(
    mu_a: Tensor,
    sigma_a: Tensor,
    mu_b: Tensor,
    sigma_b: Tensor,
    eps: float = 1e-6,
) -> float:
    """Frechet distance between two Gaussians: ``|mu_a - mu_b|^2 + tr(S_a + S_b - 2 sqrt(S_a S_b))``."""
    mu_a = mu_a.to(torch.float64).flatten()
    mu_b = mu_b.to(torch.float64).flatten()
    sigma_a = sigma_a.to(torch.float64)
    sigma_b = sigma_b.to(torch.float64)
    if mu_a.shape != mu_b.shape:
        raise ValueError("Feature dimensions must match between the two sets.")
    if sigma_a.shape != (mu_a.numel(), mu_a.numel()) or sigma_b.shape != (
        mu_b.numel(),
        mu_b.numel(),
    ):
        raise ValueError("Covariances must be (D, D) matching the mean dimensions.")
    mean_delta = float(mu_a.sub(mu_b).pow(2).sum())
    # Symmetrize and regularize before the matrix square root; the sqrtm of a
    # near-singular product is the classic source of NaN/complex FID values.
    product = sigma_a @ sigma_b
    product = (product + product.T) / 2.0
    product += eps * torch.eye(product.shape[0], dtype=product.dtype)
    try:
        from scipy.linalg import sqrtm
    except ImportError as error:  # pragma: no cover - scipy is a core dep
        raise ImportError("frechet_distance requires scipy.") from error
    sqrt_product = sqrtm(product.numpy())
    if torch.is_tensor(sqrt_product):
        sqrt_product = sqrt_product.numpy()  # pragma: no cover
    sqrt_product = sqrt_product.real
    trace_term = float(torch.diagonal(sigma_a + sigma_b).sum()) - 2.0 * float(sqrt_product.trace())
    return max(0.0, mean_delta + trace_term)


def _moment_statistics(features: Tensor) -> tuple[Tensor, Tensor]:
    features = features.to(torch.float64)
    if features.shape[0] < 2:
        raise ValueError("FID needs at least 2 images per set.")
    mu = features.mean(dim=0)
    centered = features - mu
    sigma = centered.T @ centered / (features.shape[0] - 1)
    return mu, sigma


def fid_score(images_a: Tensor, images_b: Tensor, extractor: FeatureExtractor) -> float:
    """FID between two image sets through a shared feature extractor."""
    features_a = extractor.extract(images_a)
    features_b = extractor.extract(images_b)
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError("Feature extractors must produce matching dimensions.")
    mu_a, sigma_a = _moment_statistics(features_a)
    mu_b, sigma_b = _moment_statistics(features_b)
    return frechet_distance(mu_a, sigma_a, mu_b, sigma_b)


def lpips_paired(images_a: Tensor, images_b: Tensor, net: str = "alex") -> float:
    """Mean LPIPS between elementwise-paired image batches, both in [0, 1]."""
    if images_a.shape != images_b.shape:
        raise ValueError(
            f"LPIPS is a paired metric; shapes differ: {tuple(images_a.shape)} vs {tuple(images_b.shape)}."
        )
    try:
        import lpips as lpips_module
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "lpips_paired requires the lpips package. Install it with: pip install lpips"
        ) from error
    loss_fn = lpips_module.LPIPS(net=net, verbose=False)
    # lpips expects [-1, 1]
    a = images_a.float() * 2.0 - 1.0
    b = images_b.float() * 2.0 - 1.0
    with torch.no_grad():
        values = loss_fn(a, b).flatten()
    return float(values.mean())


def clip_score(images: Tensor, prompts: Sequence[str], model_name: str = "ViT-B-32") -> float:
    """Mean CLIP prompt-alignment score, ``2.5 * max(cos, 0)`` per image."""
    if images.shape[0] != len(prompts):
        raise ValueError("clip_score is paired: one prompt per image.")
    try:
        import open_clip
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise ImportError(
            "clip_score requires open_clip_torch. Install it with: pip install open_clip_torch"
        ) from error
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, -1, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, -1, 1, 1)
    resized = F.interpolate(images.float(), size=(224, 224), mode="bilinear", align_corners=False)
    normalized = ((resized - mean) / std).to(device)
    tokens = tokenizer(list(prompts)).to(device)
    with torch.no_grad():
        image_features = F.normalize(model.encode_image(normalized).float(), dim=-1)
        text_features = F.normalize(model.encode_text(tokens).float(), dim=-1)
    cosine = (image_features * text_features).sum(dim=-1)
    return float((2.5 * cosine.clamp(min=0.0)).mean())


@dataclass(frozen=True)
class QualityReport:
    """Paired quality comparison between a base and an edited model."""

    fid: float
    lpips: float
    clip_score_base: float
    clip_score_edited: float
    num_images: int
    wall_clock_sec: float

    @property
    def clip_score_delta(self) -> float:
        return self.clip_score_base - self.clip_score_edited

    def to_dict(self) -> Dict[str, float]:
        return {
            "fid": self.fid,
            "lpips": self.lpips,
            "clip_score_base": self.clip_score_base,
            "clip_score_edited": self.clip_score_edited,
            "clip_score_delta": self.clip_score_delta,
            "num_images": float(self.num_images),
            "wall_clock_sec": self.wall_clock_sec,
        }


class QualityEvaluator:
    """Runs the paired FID / LPIPS / CLIP-score protocol on neutral prompts."""

    def __init__(
        self,
        base_adapter: GenerationAdapter,
        edited_adapter: GenerationAdapter,
        neutral_prompts: Sequence[str],
        config: ValidatorConfig = ValidatorConfig(),
        feature_extractor: Optional[FeatureExtractor] = None,
        lpips_net: str = "alex",
    ) -> None:
        self._base = base_adapter
        self._edited = edited_adapter
        self._prompts = list(neutral_prompts)
        self._config = config
        self._extractor = feature_extractor or RandomProjectionFeatures()
        self._lpips_net = lpips_net

    def run(self, seed: int) -> QualityReport:
        """Generate paired image sets for both models and compute all metrics.

        Both models use the same prompts and run-level seed; per-image seeds
        are derived deterministically, so the sets are paired. Images are
        cached via the shared ``ValidatorConfig.cache_dir`` when set.
        """
        started = time.perf_counter()
        images_base = _generate_with_config(self._base, self._prompts, seed, self._config)
        images_edited = _generate_with_config(self._edited, self._prompts, seed, self._config)
        if images_base.shape != images_edited.shape:
            raise ValueError(
                "Paired generation produced mismatched shapes "
                f"{tuple(images_base.shape)} vs {tuple(images_edited.shape)}."
            )
        fid = fid_score(images_base, images_edited, self._extractor)
        lpips_value = lpips_paired(images_base, images_edited, net=self._lpips_net)
        clip_base = clip_score(images_base, self._prompts)
        clip_edited = clip_score(images_edited, self._prompts)
        return QualityReport(
            fid=fid,
            lpips=lpips_value,
            clip_score_base=clip_base,
            clip_score_edited=clip_edited,
            num_images=len(self._prompts),
            wall_clock_sec=time.perf_counter() - started,
        )
