"""Few-step fine-tuning recovery baseline (LoRA or full), step-budget sweep.

Recovers the erased concept by fine-tuning the model on a small set of
concept images, with ``budget["steps"]`` as the sweep parameter so the
recovery-vs-cost curve can be plotted against the one-shot Heretic-DiT edit.

The training loop lives in a pluggable :class:`FinetuneBackend`
(``training_backend.py``): tests use :class:`DummyFinetuneBackend`, real runs
use :class:`DiffusersLoRABackend` / :class:`DiffusersFullFinetuneBackend`
(requires ``diffusers``; LoRA additionally requires ``peft``).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from heretic_dit.baselines.common import CostTracker, merge_cost, require_budget_key
from heretic_dit.baselines.training_backend import TrainingOutcome
from heretic_dit.interfaces import RecoveryResult

__all__ = [
    "FinetuneRecovery",
    "DummyFinetuneBackend",
    "DiffusersLoRABackend",
    "DiffusersFullFinetuneBackend",
]


class FinetuneRecovery:
    """Runner: fine-tune for ``steps`` steps, then validate generative recovery.

    Budget keys:
        validator, adapter_factory -- shared (see baselines.common).
        steps:       optimization steps (the sweep parameter; default 50).
        lr:          learning rate (default 1e-4).
        rank:        LoRA rank (LoRA backend only; default 4).
        backend:     FinetuneBackend (default: Dummy for smoke tests).
        training_prompts: prompt list describing the concept images.
        seed, num_samples: shared.
    """

    method_name = "finetune-recovery"

    DEFAULT_TRAINING_PROMPTS = [
        "a photo of a {concept}",
        "a photo of the {concept}",
        "an image of a {concept}",
        "a close-up photo of a {concept}",
        "a picture of a {concept}",
    ]

    def __init__(self, mode: str = "lora") -> None:
        if mode not in ("lora", "full"):
            raise ValueError(f"mode must be 'lora' or 'full'; got {mode!r}.")
        self._mode = mode
        self.method_name = f"{mode}-finetune-recovery"

    def run(self, erased_model: Any, concept: str, budget: Dict[str, Any]) -> RecoveryResult:
        validator = require_budget_key(budget, "validator", self.method_name)
        adapter_factory = require_budget_key(budget, "adapter_factory", self.method_name)
        backend = budget.get("backend") or DummyFinetuneBackend()
        steps = int(budget.get("steps", 50))
        lr = float(budget.get("lr", 1e-4))
        rank = int(budget.get("rank", 4))
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
            outcome = backend.train(erased_model, concept, prompts, steps, lr, rank, seed)
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
        cost["mode"] = self._mode
        cost["training_images"] = int(outcome.sample_count)
        return merge_cost(result, cost)


class DummyFinetuneBackend:
    """No-op backend for CPU tests: returns the model unchanged."""

    def train(
        self,
        model: Any,
        concept: str,
        prompts: List[str],
        steps: int,
        lr: float,
        rank: int,
        seed: int,
    ) -> TrainingOutcome:
        torch.manual_seed(seed)
        return TrainingOutcome(
            recovered_model=model,
            trainable_params=rank * 64,
            sample_count=len(prompts),
            peak_vram_mb=0.0,
            notes={"backend": "dummy", "steps": steps, "rank": rank},
        )


class _DiffusersFinetuneMixin:
    """Shared plumbing for the diffusers-based fine-tune backends."""

    _train_unet = True

    def _prepare_model(self, model: Any) -> Any:
        try:
            import diffusers  # noqa: F401 -- availability check only
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ImportError(
                f"{type(self).__name__} requires diffusers. "
                "Install it with: pip install diffusers"
            ) from error
        return model

    def _few_shot_images(self, prompts: List[str], seed: int, resolution: int) -> torch.Tensor:
        import numpy as np

        if self._images is not None:
            images = self._images
        else:
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
            images = torch.stack(tensors)
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("Few-shot images must be a non-empty (N, C, H, W) batch.")
        return images

    def _train_loop(
        self,
        model: Any,
        prompts: List[str],
        steps: int,
        lr: float,
        seed: int,
        trainable_params: torch.nn.ParameterList,
        num_train_timesteps: int,
    ) -> int:
        """Shared noise-prediction fine-tuning loop; returns steps completed."""
        import torch.nn.functional as F

        tokenizer = model.tokenizer
        text_encoder = model.text_encoder
        unet = model.unet
        scheduler = model.scheduler
        device = next(unet.parameters()).device

        generator = torch.Generator().manual_seed(seed)
        images = self._few_shot_images(prompts, seed, resolution=int(getattr(model, "_resolution", 512)))
        unet.train()
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=self._weight_decay)

        latents_batch = F.interpolate(
            images.to(device=device, dtype=torch.float32),
            size=(images.shape[-2] // 8, images.shape[-1] // 8),
            mode="bilinear",
            align_corners=False,
        )
        alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        text_cache: Dict[str, torch.Tensor] = {}
        trained_steps = 0
        with torch.enable_grad():
            for step_index in range(steps):
                image = latents_batch[step_index % latents_batch.shape[0]].unsqueeze(0)
                prompt = prompts[step_index % len(prompts)]
                if prompt not in text_cache:
                    input_ids = tokenizer(
                        [prompt], padding="max_length", truncation=True, return_tensors="pt"
                    ).input_ids.to(device)
                    with torch.no_grad():
                        text_cache[prompt] = text_encoder(input_ids)[0]
                encoder_hidden_states = text_cache[prompt]

                noise = torch.randn(image.shape, generator=generator).to(image.device)
                timestep = int(torch.randint(0, num_train_timesteps, (1,), generator=generator))
                noisy = scheduler.add_noise(image, noise, torch.tensor([timestep], device=device))
                alpha_prod = alphas_cumprod[timestep]
                target = (noisy - float(alpha_prod**0.5) * image) / float(
                    (1.0 - alpha_prod) ** 0.5
                )
                predicted = unet(
                    noisy.to(unet.dtype),
                    torch.tensor([timestep], device=device),
                    encoder_hidden_states=encoder_hidden_states.to(unet.dtype),
                ).sample
                loss = F.mse_loss(predicted.float(), target.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                trained_steps += 1
        unet.eval()
        return trained_steps


class DiffusersLoRABackend(_DiffusersFinetuneMixin):
    """LoRA fine-tuning of the UNet's cross-attention layers (requires peft)."""

    def __init__(
        self,
        images: Any = None,
        sample_fn: Any = None,
        target_modules: tuple[str, ...] = ("to_k", "to_v"),
        weight_decay: float = 0.0,
    ) -> None:
        if (images is None) == (sample_fn is None):
            raise ValueError("Provide exactly one of `images` or `sample_fn`.")
        self._images = images
        self._sample_fn = sample_fn
        self._target_modules = target_modules
        self._weight_decay = weight_decay

    def train(
        self,
        model: Any,
        concept: str,
        prompts: List[str],
        steps: int,
        lr: float,
        rank: int,
        seed: int,
    ) -> TrainingOutcome:
        model = self._prepare_model(model)
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ImportError(
                "DiffusersLoRABackend requires peft. Install it with: pip install peft"
            ) from error

        unet = model.unet
        config = LoraConfig(
            r=int(rank),
            lora_alpha=int(rank),
            target_modules=list(self._target_modules),
            lora_dropout=0.0,
        )
        peft_unet = get_peft_model(unet, config)
        trainable = torch.nn.ParameterList(
            [p for p in peft_unet.parameters() if p.requires_grad]
        )
        num_train_timesteps = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
        trained_steps = self._train_loop(model, prompts, steps, lr, seed, trainable, num_train_timesteps)
        # Keep the PEFT-wrapped unet attached so adapter weights survive.
        model.unet = peft_unet
        return TrainingOutcome(
            recovered_model=model,
            trainable_params=sum(int(p.numel()) for p in trainable),
            sample_count=len(prompts),
            notes={
                "backend": "diffusers-lora",
                "steps": trained_steps,
                "rank": rank,
                "target_modules": list(self._target_modules),
            },
        )


class DiffusersFullFinetuneBackend(_DiffusersFinetuneMixin):
    """Full fine-tuning of the UNet (memory-heavy; kept for the cost curve)."""

    def __init__(
        self,
        images: Any = None,
        sample_fn: Any = None,
        weight_decay: float = 0.0,
    ) -> None:
        if (images is None) == (sample_fn is None):
            raise ValueError("Provide exactly one of `images` or `sample_fn`.")
        self._images = images
        self._sample_fn = sample_fn
        self._weight_decay = weight_decay

    def train(
        self,
        model: Any,
        concept: str,
        prompts: List[str],
        steps: int,
        lr: float,
        rank: int,
        seed: int,
    ) -> TrainingOutcome:
        model = self._prepare_model(model)
        unet = model.unet
        trainable = torch.nn.ParameterList([p for p in unet.parameters() if p.requires_grad])
        for parameter in trainable:
            parameter.requires_grad_(True)
        num_train_timesteps = int(getattr(model.scheduler.config, "num_train_timesteps", 1000))
        trained_steps = self._train_loop(model, prompts, steps, lr, seed, trainable, num_train_timesteps)
        return TrainingOutcome(
            recovered_model=model,
            trainable_params=sum(int(p.numel()) for p in trainable),
            sample_count=len(prompts),
            notes={"backend": "diffusers-full-finetune", "steps": trained_steps},
        )
