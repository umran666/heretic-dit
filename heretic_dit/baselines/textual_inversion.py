"""Textual-inversion recovery baseline (Pham et al., 2023).

Learns a pseudo-token embedding for the erased concept from a handful of
concept images / prompts, following Pham et al., "Circumventing Concept
Erasure Methods for Text-to-Image Generative Models": the recovered concept
is re-introduced through a *learned token*, leaving model weights untouched.

The training loop lives in a pluggable :class:`TextualInversionBackend`:
tests use :class:`DummyTextualInversionBackend`, real runs use
:class:`DiffusersTextualInversionBackend` (requires ``diffusers``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from heretic_dit.baselines.common import CostTracker, merge_cost, require_budget_key
from heretic_dit.baselines.training_backend import TrainingOutcome
from heretic_dit.interfaces import RecoveryResult

__all__ = ["TextualInversionRecovery", "DummyTextualInversionBackend", "DiffusersTextualInversionBackend"]


class TextualInversionRecovery:
    """Runner: learn a pseudo-token, then validate generative recovery.

    Budget keys:
        validator, adapter_factory -- shared (see baselines.common).
        steps:       optimization steps (sweep parameter; default 300).
        lr:          embedding learning rate (default 5e-4).
        training_prompts: few-shot prompt list for the pseudo-token
                     (default: 5 simple concept prompts).
        backend:     TextualInversionBackend (default: Dummy for smoke tests).
        seed, num_samples: shared.
    """

    method_name = "textual-inversion"

    DEFAULT_TRAINING_PROMPTS = [
        "a photo of a {concept}",
        "a photo of the {concept}",
        "a close-up photo of a {concept}",
        "an image of a {concept}",
        "a picture of a {concept}",
    ]

    def run(self, erased_model: Any, concept: str, budget: Dict[str, Any]) -> RecoveryResult:
        validator = require_budget_key(budget, "validator", self.method_name)
        adapter_factory = require_budget_key(budget, "adapter_factory", self.method_name)
        backend = budget.get("backend") or DummyTextualInversionBackend()
        steps = int(budget.get("steps", 300))
        lr = float(budget.get("lr", 5e-4))
        seed = int(budget.get("seed", 0))
        if steps <= 0:
            raise ValueError(f"steps must be positive; got {steps}.")
        prompts = list(
            budget.get(
                "training_prompts",
                [t.format(concept=concept) for t in self.DEFAULT_TRAINING_PROMPTS],
            )
        )

        with CostTracker() as tracker:
            outcome = backend.train_token(erased_model, concept, prompts, steps, lr, seed)
            adapter = adapter_factory(outcome.recovered_model)
            result = validator.validate(
                adapter,
                concept,
                int(budget.get("num_samples", 16)),
                seed,
            )
        result.method = self.method_name
        cost = tracker.snapshot(
            trainable_params=outcome.trainable_params,
            sample_count=outcome.sample_count + int(result.cost.get("sample_count", 0)),
        )
        cost["peak_vram_mb"] = max(cost["peak_vram_mb"], outcome.peak_vram_mb)
        cost["steps"] = steps
        cost["lr"] = lr
        cost["training_images"] = int(outcome.sample_count)
        return merge_cost(result, cost)


class DummyTextualInversionBackend:
    """No-op backend for CPU tests: returns the model unchanged.

    Reports a plausible trainable-param count (one embedding vector of 768
    entries) so cost-accounting paths are exercised end to end.
    """

    def train_token(
        self,
        model: Any,
        concept: str,
        prompts: List[str],
        steps: int,
        lr: float,
        seed: int,
    ) -> TrainingOutcome:
        torch.manual_seed(seed)
        return TrainingOutcome(
            recovered_model=model,
            trainable_params=768,
            sample_count=len(prompts),
            peak_vram_mb=0.0,
            notes={"backend": "dummy", "steps": steps},
        )


class DiffusersTextualInversionBackend:
    """Real textual inversion against a diffusers pipeline (lazy imports).

    ``model`` must be a diffusers pipeline (or a compatible object) exposing
    ``tokenizer``, ``text_encoder``, ``unet``, and ``scheduler``. Only the
    placeholder-token embedding row is trained, with the standard
    noise-prediction MSE objective, following the official diffusers
    textual-inversion recipe and Pham et al.'s few-shot recovery usage.

    Few-shot images come from either:
        * ``images`` passed at construction (``(N, C, H, W)`` in [0, 1]), or
        * a ``sample_fn(prompt, seed) -> Tensor`` used to synthesize them
          from the prompts (typically the *unerased* reference model).
    """

    placeholder_template = "<{concept}>"

    def __init__(
        self,
        images: Any = None,
        sample_fn: Any = None,
        batch_size: int = 2,
        resolution: int = 512,
        weight_decay: float = 0.0,
    ) -> None:
        if (images is None) == (sample_fn is None):
            raise ValueError("Provide exactly one of `images` or `sample_fn`.")
        self._images = images
        self._sample_fn = sample_fn
        self._batch_size = batch_size
        self._resolution = resolution
        self._weight_decay = weight_decay

    def _few_shot_images(self, prompts: List[str], seed: int) -> Any:
        import numpy as np

        if self._images is not None:
            return self._images
        tensors: List[torch.Tensor] = []
        for index, prompt in enumerate(prompts):
            image = self._sample_fn(prompt, seed + index)
            if isinstance(image, torch.Tensor):
                tensor = image
                if tensor.ndim == 4:
                    tensor = tensor[0]
            else:
                array = np.asarray(image).astype("float32") / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor)
        return torch.stack(tensors)

    def train_token(
        self,
        model: Any,
        concept: str,
        prompts: List[str],
        steps: int,
        lr: float,
        seed: int,
    ) -> TrainingOutcome:
        tokenizer = model.tokenizer
        text_encoder = model.text_encoder
        unet = model.unet
        scheduler = model.scheduler

        placeholder = self.placeholder_template.format(concept=concept)
        num_added = tokenizer.add_tokens([placeholder])
        if num_added != 1:
            raise RuntimeError(f"Failed to add placeholder token {placeholder!r}.")
        token_id = tokenizer.convert_tokens_to_ids(placeholder)

        embedding_layer = text_encoder.get_input_embeddings()
        embedding_dtype = embedding_layer.weight.dtype
        embedding_dim = embedding_layer.weight.shape[1]
        device = embedding_layer.weight.device

        generator = torch.Generator().manual_seed(seed)
        new_row = torch.randn(1, embedding_dim, generator=generator).to(
            device=device, dtype=embedding_dtype
        )
        embedding_layer.weight.data = torch.cat([embedding_layer.weight.data, new_row], dim=0)
        new_index = embedding_layer.weight.shape[0] - 1

        trainable = torch.nn.Parameter(embedding_layer.weight.data[new_index : new_index + 1])
        optimizer = torch.optim.AdamW(
            [trainable], lr=lr, weight_decay=self._weight_decay
        )
        embedding_layer.weight.requires_grad_(False)
        embedding_layer.weight.data[new_index : new_index + 1] = trainable.data

        images = self._few_shot_images(prompts, seed)
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("Few-shot images must be a non-empty (N, C, H, W) batch.")
        images = images.to(device=device, dtype=unet.dtype)

        # Freeze everything except the learned row; unet stays frozen.
        for parameter in unet.parameters():
            parameter.requires_grad_(False)
        for parameter in text_encoder.parameters():
            parameter.requires_grad_(False)

        num_train_timesteps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
        prompts_cycle = [prompts[i % len(prompts)] for i in range(steps)]
        image_cycle = [images[i % images.shape[0]] for i in range(steps)]
        alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

        unet.eval()
        text_encoder.eval()
        trained_steps = 0
        with torch.enable_grad():
            for step_index in range(steps):
                prompt = prompts_cycle[step_index]
                image = image_cycle[step_index]
                input_ids = tokenizer(
                    [prompt],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)

                # Insert the trainable embedding row in place of the
                # placeholder position without rebuilding the whole table.
                base_embeddings = text_encoder.input_embeds_with_placeholder(
                    input_ids, new_index, trainable.to(embedding_dtype)
                ) if hasattr(text_encoder, "input_embeds_with_placeholder") else None
                if base_embeddings is None:
                    token_embeddings = embedding_layer.weight.data[input_ids].clone()
                    mask = input_ids == token_id
                    token_embeddings[mask] = trainable.to(embedding_dtype).reshape(-1)
                    base_embeddings = token_embeddings
                encoder_hidden_states = text_encoder(
                    inputs_embeds=base_embeddings.to(next(text_encoder.parameters()).dtype),
                    attention_mask=None,
                )[0]

                latents = F.interpolate(
                    image.unsqueeze(0), size=(self._resolution // 8, self._resolution // 8),
                    mode="bilinear", align_corners=False,
                ).to(torch.float32)
                noise = torch.randn(latents.shape, generator=generator).to(latents.device)
                timestep = int(torch.randint(0, num_train_timesteps, (1,), generator=generator))
                noisy = scheduler.add_noise(latents, noise, torch.tensor([timestep], device=device))

                alpha_prod = alphas_cumprod[timestep]
                sqrt_alpha = float(alpha_prod**0.5)
                sqrt_one_minus = float((1.0 - alpha_prod) ** 0.5)
                predicted = unet(
                    noisy.to(unet.dtype),
                    torch.tensor([timestep], device=device),
                    encoder_hidden_states=encoder_hidden_states.to(unet.dtype),
                ).sample
                # eps-prediction target: eps = (x_t - sqrt(alpha)*x_0) / sqrt(1-alpha).
                target = (noisy - sqrt_alpha * latents) / sqrt_one_minus
                loss = F.mse_loss(predicted.float(), target.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                embedding_layer.weight.data[new_index : new_index + 1] = trainable.data
                trained_steps += 1

        return TrainingOutcome(
            recovered_model=model,
            trainable_params=int(trainable.numel()),
            sample_count=int(images.shape[0]),
            notes={
                "backend": "diffusers-textual-inversion",
                "steps": trained_steps,
                "placeholder": placeholder,
                "lr": lr,
                "seed": seed,
            },
        )
