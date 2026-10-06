"""Modal experiment comparing Heretic-DiT against Fine-Tuning and Textual Inversion baselines.

Empirical evaluation protocol:
1. Base model: CompVis/stable-diffusion-v1-4 (pinned).
2. Erased model: Clean ESD-x control (300 steps, lr=5e-5, Delta W non-cross-attn == 0.0).
3. Evaluates 5 approaches under the identical multi-class ClipStyleScorer:
   - Base SD 1.4 (unerased upper bound)
   - Clean ESD-x (zero-shot erased lower bound)
   - Heretic-DiT (training-free rank-1 closed-form projection, 0 steps, 0 trainable params)
   - LoRA Fine-Tuning (cross-attn rank-4, 50 and 150 steps)
   - Textual Inversion (pseudo-token <van-gogh>, 100 and 300 steps)
4. Metrics:
   - Training wall-clock time (seconds)
   - Peak VRAM (MB)
   - Trainable parameters
   - Multi-class Van Gogh style score S_vg on held-out concept prompts
   - Normalized concept recovery: (S - S_esd) / (S_base - S_esd)
   - Neutral style bleed S_neutral on neutral control prompts
   - Specificity delta: S_vg - S_neutral
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("heretic-dit-baselines-comparison")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0",
        "diffusers>=0.28.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "safetensors>=0.4.0",
        "peft>=0.10.0",
        "optuna>=3.6.0",
        "scipy>=1.12.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "open_clip_torch>=2.24.0",
        "torchvision>=0.17.0",
    )
    .add_local_python_source("heretic_dit")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("splits", remote_path="/root/splits")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
)
def run_baselines_benchmark() -> Dict[str, Any]:
    """Execute end-to-end baseline comparison on Modal."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
    from heretic_dit.baselines.finetune_recovery import DiffusersLoRABackend
    from heretic_dit.baselines.textual_inversion import DiffusersTextualInversionBackend
    from heretic_dit.benchmarks.concepts import load_or_create_split
    from heretic_dit.core.subspace import mean_difference
    from heretic_dit.eval.classifiers import ClipStyleScorer
    from heretic_dit.search.editing import LayerEdit, applied_edit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Modal Baselines Benchmark starting on {torch.cuda.get_device_name(0)} ===")

    # -------------------------------------------------------------------------
    # 1. Load Base SD 1.4 Pipeline
    # -------------------------------------------------------------------------
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading base pipeline from {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae

    # Store CPU copies of base weights to preserve GPU VRAM
    base_unet_state_dict = {k: v.cpu().clone() for k, v in unet.state_dict().items()}
    clean_text_encoder = copy.deepcopy(text_encoder).to("cpu")
    clean_tokenizer = copy.deepcopy(tokenizer)

    # -------------------------------------------------------------------------
    # 2. Train Clean ESD-x Control Checkpoint
    # -------------------------------------------------------------------------
    print("\n--- Training Clean ESD-x Control (300 steps) ---")
    torch.manual_seed(42)
    np.random.seed(42)

    esd_prompts = [
        "a painting in the style of Van Gogh",
        "a portrait in the style of Van Gogh",
        "a landscape in the style of Van Gogh",
        "starry night painting by Van Gogh",
        "sunflowers painting by Van Gogh",
        "oil painting by Van Gogh",
        "an artwork by Van Gogh",
        "canvas painting in the style of Van Gogh",
    ]

    # Pre-sample 16 clean latent trajectories
    precomputed_trajectories = []
    print("Pre-sampling 16 clean reference trajectories...")
    for idx in range(16):
        prompt = esd_prompts[idx % len(esd_prompts)]
        seed = 1000 + idx
        g = torch.Generator(device).manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=g, device=device)
        text_inputs = tokenizer([prompt], padding="max_length", max_length=77, return_tensors="pt")
        text_emb = text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_inputs = tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
        uncond_emb = text_encoder(uncond_inputs.input_ids.to(device))[0]

        pipe.scheduler.set_timesteps(25, device=device)
        for t in pipe.scheduler.timesteps:
            with torch.no_grad():
                pred_pos = unet(latents, t, encoder_hidden_states=text_emb).sample
                pred_neg = unet(latents, t, encoder_hidden_states=uncond_emb).sample
            precomputed_trajectories.append({
                "t": t.item(),
                "latents": latents.clone().detach(),
                "text_emb": text_emb.clone().detach(),
                "pred_pos": pred_pos.clone().detach(),
                "pred_neg": pred_neg.clone().detach(),
            })
            pred_cfg = pred_neg + 7.5 * (pred_pos - pred_neg)
            latents = pipe.scheduler.step(pred_cfg, t, latents).prev_sample

    print(f"Precomputed {len(precomputed_trajectories)} clean trajectory states.")

    # Select trainable cross-attention parameters
    trainable_params = []
    for name, param in unet.named_parameters():
        if "attn2.to_k" in name or "attn2.to_v" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-2)
    esd_start_time = time.perf_counter()

    for step in range(300):
        optimizer.zero_grad()
        sample = random.choice(precomputed_trajectories)
        t_val = torch.tensor([sample["t"]], device=device)
        latents_val = sample["latents"]
        text_emb_val = sample["text_emb"]

        # ESD cross-entropy negative guidance target
        e_0 = sample["pred_neg"]
        e_p = sample["pred_pos"]
        target = e_0 - 1.0 * (e_p - e_0)

        pred_new = unet(latents_val, t_val, encoder_hidden_states=text_emb_val).sample
        loss = F.mse_loss(pred_new, target)
        loss.backward()
        optimizer.step()

    esd_train_time = time.perf_counter() - esd_start_time
    print(f"Clean ESD-x trained in {esd_train_time:.2f}s.")

    # Save clean ESD state dict to CPU
    clean_esd_unet_state_dict = {k: v.cpu().clone() for k, v in unet.state_dict().items()}

    # Verify bit-identity of non-cross-attn weights
    for name, p_esd in clean_esd_unet_state_dict.items():
        if "attn2.to_k" not in name and "attn2.to_v" not in name:
            p_base = base_unet_state_dict[name]
            diff = (p_esd - p_base).abs().max().item()
            assert diff == 0.0, f"Leaked diff in {name}: {diff}"
    print("Bit-identity verified: 100% non-cross-attn parameters have diff == 0.0.")

    # Free training memory immediately
    del optimizer, precomputed_trajectories, trainable_params
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. Setup Multi-Class ClipStyleScorer and Test Prompts
    # -------------------------------------------------------------------------
    scorer = ClipStyleScorer(model_name="ViT-B-32", pretrained="openai", device=str(device))

    test_concept_prompts = [
        "a painting of starry night over the rhone in the style of Van Gogh",
        "a portrait of a postman in the style of Van Gogh",
        "sunflowers in a vase by Van Gogh",
        "a wheatfield with cypresses by Van Gogh",
        "an olive tree grove in the style of Van Gogh",
        "a cafe terrace at night painted by Van Gogh",
        "an impressionist bedroom in arles by Van Gogh",
        "an old church in auvers painted by Van Gogh",
    ]

    test_neutral_prompts = [
        "a photograph of a golden retriever playing in a park",
        "a modern kitchen with marble countertops and steel appliances",
        "a commercial airliner flying through cloudy blue sky",
        "a fresh red apple sitting on a rustic wooden dining table",
        "a red sports car parked in front of a modern glass skyscraper",
        "a snowy pine forest in winter morning sunlight",
        "a close-up portrait of an elderly man with glasses in black and white",
        "a calm lake reflecting green mountain hills at sunrise",
    ]

    def evaluate_model(
        pipeline: StableDiffusionPipeline,
        concept_prompts: List[str],
        neutral_prompts: List[str],
        tag: str,
    ) -> Dict[str, float]:
        """Generate test images and compute style & neutral bleed scores."""
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        concept_imgs = []
        for i, p in enumerate(concept_prompts):
            g = torch.Generator(device).manual_seed(42 + i)
            with torch.no_grad():
                img = pipeline(p, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
            arr = np.array(img).astype(np.float32) / 255.0
            concept_imgs.append(torch.from_numpy(arr).permute(2, 0, 1))

        neutral_imgs = []
        for i, p in enumerate(neutral_prompts):
            g = torch.Generator(device).manual_seed(100 + i)
            with torch.no_grad():
                img = pipeline(p, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
            arr = np.array(img).astype(np.float32) / 255.0
            neutral_imgs.append(torch.from_numpy(arr).permute(2, 0, 1))

        scores_concept = scorer.classify(concept_imgs, "van gogh")
        scores_neutral = scorer.classify(neutral_imgs, "van gogh")
        s_concept = float(np.mean(scores_concept))
        s_neutral = float(np.mean(scores_neutral))
        print(f"[{tag}] Van Gogh Score: {s_concept:.4f} | Neutral Bleed: {s_neutral:.4f}")
        return {
            "s_concept": s_concept,
            "s_neutral": s_neutral,
            "specificity_gap": float(s_concept - s_neutral),
        }

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Baseline A: Base SD 1.4 (Unerased Upper Bound)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Base SD 1.4 (Unerased Reference) ---")
    pipe.unet.load_state_dict(base_unet_state_dict)
    base_res = evaluate_model(pipe, test_concept_prompts, test_neutral_prompts, "Base SD 1.4")
    results["base_sd14"] = {
        "method": "Base SD 1.4 (Reference)",
        "wall_clock_sec": 0.0,
        "trainable_params": 0,
        "peak_vram_mb": 0.0,
        **base_res,
        "normalized_recovery": 1.0,
    }

    # -------------------------------------------------------------------------
    # Baseline B: Clean ESD-x (Zero-Shot Erased Lower Bound)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Clean ESD-x (Zero-Shot Erased) ---")
    pipe.unet.load_state_dict(clean_esd_unet_state_dict)
    clean_esd_res = evaluate_model(pipe, test_concept_prompts, test_neutral_prompts, "Clean ESD-x")
    results["clean_esd"] = {
        "method": "Clean ESD-x (Erased)",
        "wall_clock_sec": 0.0,
        "trainable_params": 0,
        "peak_vram_mb": 0.0,
        **clean_esd_res,
        "normalized_recovery": 0.0,
    }

    gap = base_res["s_concept"] - clean_esd_res["s_concept"]
    print(f"Empirical erasure gap: {gap:.4f} ({clean_esd_res['s_concept']:.4f} -> {base_res['s_concept']:.4f})")

    # -------------------------------------------------------------------------
    # Baseline C: Heretic-DiT (Ours - Training-Free Rank-1 Closed-Form Edit)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Heretic-DiT (Training-Free Closed-Form Projection) ---")
    pipe.unet.load_state_dict(clean_esd_unet_state_dict)
    torch.cuda.reset_peak_memory_stats()
    heretic_start = time.perf_counter()

    # Compute contrastive direction from positive concept vs neutral prompts
    split = load_or_create_split(config_path=Path("/root/configs/concepts_default.yaml"), split_dir=Path("/root/splits"))
    neutral_train_prompts = list(split.registry.neutral_prompts[:8])
    vg_train_prompts = list(split.registry.get("van gogh").search_prompts[:8])

    p_tok = tokenizer(vg_train_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    n_tok = tokenizer(neutral_train_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        p_emb = text_encoder(p_tok)[0]
        n_emb = text_encoder(n_tok)[0]
    v_vg = mean_difference(p_emb.reshape(-1, 768), n_emb.reshape(-1, 768), compute_dtype=torch.float32).to(device)

    # Apply Trial 29 best specific configuration: middle blocks [5, 10], both projections
    adapter = DiffusersModelAdapter(pipe.unet)
    k_layers = adapter.get_cross_attention_layer_names(layer_types=("key",))
    v_layers = adapter.get_cross_attention_layer_names(layer_types=("value",))
    target_names = [k_layers[i] for i in range(5, 11) if i < len(k_layers)] + [
        v_layers[i] for i in range(5, 11) if i < len(v_layers)
    ]

    active_edits = [
        LayerEdit(
            name=name,
            weight=adapter.target_layers[name].module.weight,
            alpha=0.697,
            directions=v_vg,
            mode="orthogonal",
            side="input",
        )
        for name in target_names
    ]

    with applied_edit(pipe.unet, active_edits):
        heretic_time = time.perf_counter() - heretic_start
        heretic_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        heretic_res = evaluate_model(pipe, test_concept_prompts, test_neutral_prompts, "Heretic-DiT")

    norm_rec_heretic = (heretic_res["s_concept"] - clean_esd_res["s_concept"]) / gap
    results["heretic_dit"] = {
        "method": "Heretic-DiT (Ours)",
        "wall_clock_sec": float(heretic_time),
        "trainable_params": 0,
        "peak_vram_mb": float(heretic_vram),
        **heretic_res,
        "normalized_recovery": float(norm_rec_heretic),
    }

    # Clean cache before few-shot and training baselines
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Generate 5 Few-Shot Concept Images from Base Model for Baselines
    # -------------------------------------------------------------------------
    print("\n--- Generating 5 Few-Shot Concept Images from Base Model for Baselines ---")
    few_shot_prompts = [
        "a painting of starry night by Van Gogh",
        "sunflowers in a vase painted by Van Gogh",
        "a self portrait of Vincent Van Gogh",
        "an olive trees landscape by Van Gogh",
        "a cafe at night in the style of Van Gogh",
    ]
    pipe.unet.load_state_dict(base_unet_state_dict)
    few_shot_images = []
    for idx, prompt in enumerate(few_shot_prompts):
        g = torch.Generator(device).manual_seed(200 + idx)
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
        arr = np.array(img).astype(np.float32) / 255.0
        few_shot_images.append(torch.from_numpy(arr).permute(2, 0, 1))
    few_shot_images_tensor = torch.stack(few_shot_images)

    # -------------------------------------------------------------------------
    # Baseline D: LoRA Fine-Tuning (50 and 150 steps)
    # -------------------------------------------------------------------------
    for lora_steps in [50, 150]:
        print(f"\n--- Running LoRA Fine-Tuning ({lora_steps} steps) ---")
        pipe.unet.load_state_dict(clean_esd_unet_state_dict)
        torch.cuda.reset_peak_memory_stats()

        lora_backend = DiffusersLoRABackend(images=few_shot_images_tensor)
        t_lora_start = time.perf_counter()
        outcome = lora_backend.train(
            pipe,
            concept="Van Gogh",
            prompts=few_shot_prompts,
            steps=lora_steps,
            lr=1e-4,
            rank=4,
            seed=42,
        )
        lora_wall_clock = time.perf_counter() - t_lora_start
        lora_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        lora_res = evaluate_model(pipe, test_concept_prompts, test_neutral_prompts, f"LoRA-{lora_steps}")
        norm_rec_lora = (lora_res["s_concept"] - clean_esd_res["s_concept"]) / gap

        results[f"lora_{lora_steps}"] = {
            "method": f"LoRA ({lora_steps} steps, r=4)",
            "wall_clock_sec": float(lora_wall_clock),
            "trainable_params": int(outcome.trainable_params),
            "peak_vram_mb": float(lora_peak_vram),
            **lora_res,
            "normalized_recovery": float(norm_rec_lora),
        }

        # Safely unwrap and remove PEFT adapter
        if hasattr(pipe.unet, "unload"):
            pipe.unet = pipe.unet.unload()
        elif hasattr(pipe.unet, "base_model") and hasattr(pipe.unet.base_model, "model"):
            pipe.unet = pipe.unet.base_model.model
        del outcome, lora_backend
        torch.cuda.empty_cache()
        gc.collect()

    # -------------------------------------------------------------------------
    # Baseline E: Textual Inversion (100 and 300 steps)
    # -------------------------------------------------------------------------
    import re
    for ti_steps in [100, 300]:
        print(f"\n--- Running Textual Inversion ({ti_steps} steps) ---")
        # Reset UNet to clean ESD
        pipe.unet.load_state_dict(clean_esd_unet_state_dict)
        # Restore pristine text encoder and tokenizer
        pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
        pipe.tokenizer = copy.deepcopy(clean_tokenizer)

        torch.cuda.reset_peak_memory_stats()
        ti_backend = DiffusersTextualInversionBackend(images=few_shot_images_tensor)
        t_ti_start = time.perf_counter()
        outcome_ti = ti_backend.train_token(
            pipe,
            concept="Van Gogh",
            prompts=few_shot_prompts,
            steps=ti_steps,
            lr=5e-4,
            seed=42,
        )
        ti_wall_clock = time.perf_counter() - t_ti_start
        ti_peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Substitute "Van Gogh" with learned placeholder "<Van Gogh>" in test prompts
        placeholder = outcome_ti.notes.get("placeholder", "<Van Gogh>")
        vg_pattern = re.compile(re.escape("Van Gogh"), re.IGNORECASE)
        ti_concept_prompts = [vg_pattern.sub(placeholder, p) for p in test_concept_prompts]

        ti_res = evaluate_model(pipe, ti_concept_prompts, test_neutral_prompts, f"TextualInversion-{ti_steps}")
        norm_rec_ti = (ti_res["s_concept"] - clean_esd_res["s_concept"]) / gap

        results[f"textual_inversion_{ti_steps}"] = {
            "method": f"Textual Inversion ({ti_steps} steps)",
            "wall_clock_sec": float(ti_wall_clock),
            "trainable_params": int(outcome_ti.trainable_params),
            "peak_vram_mb": float(ti_peak_vram),
            **ti_res,
            "normalized_recovery": float(norm_rec_ti),
        }

        # Reset text encoder and tokenizer to clean reference
        pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
        pipe.tokenizer = copy.deepcopy(clean_tokenizer)
        del outcome_ti, ti_backend
        torch.cuda.empty_cache()
        gc.collect()

    print("\n=== All Baselines Completed! ===")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main():
    print("Launching Modal baselines benchmark...")
    res = run_baselines_benchmark.remote()
    out_file = Path("baseline_comparison_results.json")
    out_file.write_text(json.dumps(res, indent=2))
    print(f"Results successfully saved to {out_file.absolute()}")
