"""Generation adapters bridging pipelines to the eval layer.

The eval layer never touches a concrete diffusers pipeline. Antigravity's
pipeline hooks (SD 1.5 / SDXL / DiT integration) expose their models through
the :class:`GenerationAdapter` protocol defined here: one prompt, one fixed
seed, one call, one image tensor. Anything that satisfies the protocol
(including a plain callable wrapped in :class:`CallableAdapter`) can be
validated without any diffusers dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import torch
from torch import Tensor

__all__ = [
    "GenerationAdapter",
    "CallableAdapter",
    "DiffusersAdapter",
    "deterministic_image_seeds",
]


@runtime_checkable
class GenerationAdapter(Protocol):
    """Minimal uniform interface for image generation used by all eval code.

    Contract:
        * ``generate`` is deterministic: the same ``(prompt, seed, steps,
          guidance)`` always yields the same image, on any adapter.
        * The returned tensor is float in ``[0, 1]`` with shape ``(C, H, W)``
          (a leading batch dimension of size 1 is also accepted).
        * ``model_id`` must be a stable string identifying the *weights*
          (e.g. ``"sd15-erased/esd/van-gogh"``); it enters cache keys and
          seed derivation, so it must differ between base and edited models.
    """

    model_id: str

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Tensor:
        """Generate a single image for ``prompt`` from integer ``seed``."""
        ...


@dataclass(frozen=True)
class CallableAdapter:
    """Adapts a plain callable into a :class:`GenerationAdapter`.

    ``generate_fn`` must map ``(prompt, seed, num_inference_steps,
    guidance_scale)`` to a ``(C, H, W)`` float tensor in ``[0, 1]``.
    """

    generate_fn: Callable[[str, int, int, float], Tensor]
    model_id: str

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Tensor:
        image = self.generate_fn(prompt, seed, num_inference_steps, guidance_scale)
        return _canonicalize_image(image)


def _canonicalize_image(image: Tensor) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError(f"Generation adapters must return torch.Tensor, got {type(image)!r}.")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Generated image must have shape (C, H, W); got {tuple(image.shape)}.")
    if image.dtype != torch.float32:
        image = image.to(torch.float32)
    if float(image.min()) < -1e-4 or float(image.max()) > 1.0 + 1e-4:
        raise ValueError(
            "Generated images must be in [0, 1]; "
            f"got range [{float(image.min()):.4f}, {float(image.max()):.4f}]."
        )
    return image


class DiffusersAdapter:
    """Adapter for a diffusers ``StableDiffusionXLPipeline``-style pipeline.

    Requires ``diffusers`` and ``transformers``. All generation is routed
    through a ``torch.Generator`` seeded from ``seed`` so base and edited
    models can be compared pairwise.
    """

    def __init__(
        self,
        pipeline: Any,
        model_id: str,
        height: int = 512,
        width: int = 512,
    ) -> None:
        self._pipeline = pipeline
        self.model_id = model_id
        self._height = height
        self._width = width

    def generate(
        self,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Tensor:
        import numpy as np

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        output = self._pipeline(
            prompt=prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
            height=self._height,
            width=self._width,
        )
        images = getattr(output, "images", None)
        if not images:
            raise RuntimeError("Diffusers pipeline returned no images.")
        array = np.asarray(images[0]).astype("float32") / 255.0
        return _canonicalize_image(torch.from_numpy(array).permute(2, 0, 1))


def _seed_digest(*parts: Any) -> int:
    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def deterministic_image_seeds(
    model_id: str,
    prompts: list[str],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> list[int]:
    """Derive one stable per-image integer seed from the run-level seed.

    The derivation mixes the model id, prompt, and sampler settings into the
    seed so that (a) base and edited models use *different* but reproducible
    seeds, and (b) adding a prompt never shifts the seeds of other prompts.
    """
    return [
        _seed_digest(model_id, prompt, seed, num_inference_steps, guidance_scale)
        % (2**31 - 1)
        for prompt in prompts
    ]
