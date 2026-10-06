"""Modal script: Concept Recovery Baselines (Textual Inversion & LoRA sweep) with Paired LPIPS.

Protocol:
1. Reuses pinned Clean ESD-x from Modal Volume 'heretic-models':
   - Base SD 1.4: CompVis/stable-diffusion-v1-4
   - Clean ESD-x: /root/models/clean_esd_x_sd14_van_gogh.safetensors
2. Unified Evaluation:
   - 16 held-out concept prompts + 16 neutral control prompts.
   - 3 seeds ([42, 123, 999]), total N = 48 concept and N = 48 neutral images per condition.
   - Base SD 1.4 neutral images cached for paired LPIPS and FID calculation.
3. Experiments:
   - Base SD 1.4 reference
   - Clean ESD-x erased baseline
   - Textual Inversion positive control on Base SD 1.4 (1000 steps, init='art')
   - Textual Inversion on Clean ESD-x (1000 steps, init='art')
   - LoRA Convergence Sweep (50, 150, 300, 500, 1000 steps, r=4)
   - Paired-bootstrap CIs on normalized recovery and paired LPIPS for every row!
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("heretic-baselines-ti-lora")

models_volume = modal.Volume.from_name("heretic-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0",
        "diffusers>=0.28.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "safetensors>=0.4.0",
        "peft>=0.10.0",
        "scipy>=1.12.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "open_clip_torch>=2.24.0",
        "torchvision>=0.17.0",
        "lpips>=0.1.4",
    )
    .add_local_python_source("heretic_dit")
)


def paired_bootstrap_metrics(
    s_concept_method: List[float],
    s_neutral_method: List[float],
    s_concept_base: List[float],
    s_neutral_base: List[float],
    s_concept_esd: List[float],
    s_neutral_esd: List[float],
    lpips_method: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    import numpy as np

    n = len(s_concept_method)
    indices = np.random.choice(n, size=(n_resamples, n), replace=True)

    c_m = np.array(s_concept_method, dtype=np.float64)
    n_m = np.array(s_neutral_method, dtype=np.float64)
    c_b = np.array(s_concept_base, dtype=np.float64)
    c_e = np.array(s_concept_esd, dtype=np.float64)
    lp = np.array(lpips_method, dtype=np.float64)

    c_m_means = np.mean(c_m[indices], axis=1)
    n_m_means = np.mean(n_m[indices], axis=1)
    c_b_means = np.mean(c_b[indices], axis=1)
    c_e_means = np.mean(c_e[indices], axis=1)
    lp_means = np.mean(lp[indices], axis=1)

    gaps = c_b_means - c_e_means
    recoveries = (c_m_means - c_e_means) / np.maximum(gaps, 1e-6)

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.percentile(arr, (1.0 - ci) / 2.0 * 100.0)),
            "ci_high": float(np.percentile(arr, (1.0 + ci) / 2.0 * 100.0)),
        }

    return {
        "s_concept": stats(c_m_means),
        "s_neutral": stats(n_m_means),
        "specificity_gap": float(np.mean(c_m) - np.mean(n_m)),
        "normalized_recovery": stats(recoveries),
        "lpips_paired": stats(lp_means),
    }


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/models": models_volume},
    timeout=5400,
)
def run_baselines_sweep() -> Dict[str, Any]:
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    import lpips
    from safetensors.torch import load_file

    from heretic_dit.baselines.finetune_recovery import DiffusersLoRABackend
    from heretic_dit.baselines.textual_inversion import DiffusersTextualInversionBackend
    from heretic_dit.eval.classifiers import ClipStyleScorer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Baselines Sweep (TI & LoRA) starting on {torch.cuda.get_device_name(0)} ===")
    models_dir = Path("/root/models")

    # 1. Load Base SD 1.4 Pipeline
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

    base_unet_state = {k: v.cpu().clone() for k, v in unet.state_dict().items()}
    clean_text_encoder = copy.deepcopy(text_encoder).to("cpu")
    clean_tokenizer = copy.deepcopy(tokenizer)

    # 2. Load Clean ESD-x from Volume
    clean_esd_file = models_dir / "clean_esd_x_sd14_van_gogh.safetensors"
    print(f"Loading pinned Clean ESD-x from {clean_esd_file}...")
    clean_esd_state = load_file(str(clean_esd_file))

    # 3. Setup Scorers & Evaluator
    scorer = ClipStyleScorer(model_name="ViT-B-32", pretrained="openai", device=str(device))
    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)

    concept_prompts_16 = [
        "a painting of starry night over the rhone in the style of Van Gogh",
        "a portrait of a postman in the style of Van Gogh",
        "sunflowers in a vase by Van Gogh",
        "a wheatfield with cypresses by Van Gogh",
        "an olive tree grove in the style of Van Gogh",
        "a cafe terrace at night painted by Van Gogh",
        "an impressionist bedroom in arles by Van Gogh",
        "an old church in auvers painted by Van Gogh",
        "a vibrant night sky with swirling stars by Van Gogh",
        "a self portrait with felt hat painted by Van Gogh",
        "an orchard in blossom painted by Van Gogh",
        "irises blooming in a garden by Van Gogh",
        "the yellow house in arles in the style of Van Gogh",
        "a peasant harvesting wheat painted by Van Gogh",
        "cypresses against a crescent moon by Van Gogh",
        "a view of the hospital garden in saint-remy by Van Gogh",
    ]

    neutral_prompts_16 = [
        "a photograph of a golden retriever playing in a park",
        "a modern kitchen with marble countertops and steel appliances",
        "a commercial airliner flying through cloudy blue sky",
        "a fresh red apple sitting on a rustic wooden dining table",
        "a red sports car parked in front of a modern glass skyscraper",
        "a snowy pine forest in winter morning sunlight",
        "a close-up portrait of an elderly man with glasses in black and white",
        "a calm lake reflecting green mountain hills at sunrise",
        "a crowded downtown subway station during morning commute",
        "a plate of homemade pasta with tomato basil sauce",
        "a tabby cat sleeping on a sunny windowsill",
        "a drone view of a container shipping port at sunset",
        "a ceramic coffee mug on an office desk next to a laptop",
        "a hiker standing on top of a rocky mountain summit",
        "a concrete bridge spanning across a wide river",
        "a bookshelf filled with colorful vintage hardcover novels",
    ]

    eval_seeds = [42, 123, 999]  # N = 48 images per prompt type

    def evaluate_condition(
        tag: str,
        concept_prompts: List[str] = concept_prompts_16,
        base_neutral_tensors: List[torch.Tensor] = None,
    ) -> Tuple[List[float], List[float], List[float], List[torch.Tensor]]:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        concept_scores = []
        for prompt in concept_prompts:
            imgs = []
            for seed in eval_seeds:
                g = torch.Generator(device).manual_seed(seed + (hash(prompt) % 10000))
                with torch.no_grad():
                    img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
                arr = np.array(img).astype(np.float32) / 255.0
                imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
            batch_scores = scorer.classify(imgs, "van gogh")
            concept_scores.extend(batch_scores)

        neutral_scores = []
        neutral_img_tensors = []
        for prompt in neutral_prompts_16:
            imgs = []
            for seed in eval_seeds:
                g = torch.Generator(device).manual_seed(seed + (hash(prompt) % 10000))
                with torch.no_grad():
                    img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
                arr = np.array(img).astype(np.float32) / 255.0
                tensor_img = torch.from_numpy(arr).permute(2, 0, 1)
                imgs.append(tensor_img)
                neutral_img_tensors.append(tensor_img)
            batch_scores = scorer.classify(imgs, "van gogh")
            neutral_scores.extend(batch_scores)

        lpips_scores = []
        if base_neutral_tensors is not None:
            with torch.no_grad():
                for m_img, b_img in zip(neutral_img_tensors, base_neutral_tensors):
                    m_scaled = (m_img.unsqueeze(0).to(device) * 2.0 - 1.0)
                    b_scaled = (b_img.unsqueeze(0).to(device) * 2.0 - 1.0)
                    val = lpips_alex(m_scaled, b_scaled).item()
                    lpips_scores.append(float(val))
        else:
            lpips_scores = [0.0] * len(neutral_scores)

        mean_c = float(np.mean(concept_scores))
        mean_n = float(np.mean(neutral_scores))
        mean_lp = float(np.mean(lpips_scores))
        print(f"[{tag}] S_vg: {mean_c:.4f} | S_neut: {mean_n:.4f} | LPIPS: {mean_lp:.4f} | Gap: {mean_c - mean_n:+.4f}")
        return concept_scores, neutral_scores, lpips_scores, neutral_img_tensors

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Baseline 1: Base SD 1.4 Reference
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Base SD 1.4 Reference ---")
    pipe.unet.load_state_dict(base_unet_state)
    pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
    pipe.tokenizer = copy.deepcopy(clean_tokenizer)
    c_base, n_base, lp_base, base_neut_imgs = evaluate_condition("Base SD 1.4")
    base_metrics = paired_bootstrap_metrics(c_base, n_base, c_base, n_base, c_base, n_base, lp_base)
    results["base_sd14"] = base_metrics

    # -------------------------------------------------------------------------
    # Baseline 2: Clean ESD-x Baseline
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Clean ESD-x Baseline ---")
    pipe.unet.load_state_dict(clean_esd_state)
    pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
    pipe.tokenizer = copy.deepcopy(clean_tokenizer)
    c_esd, n_esd, lp_esd, _ = evaluate_condition("Clean ESD-x", base_neutral_tensors=base_neut_imgs)
    clean_esd_metrics = paired_bootstrap_metrics(c_esd, n_esd, c_base, n_base, c_esd, n_esd, lp_esd)
    results["clean_esd"] = clean_esd_metrics

    # -------------------------------------------------------------------------
    # Few-Shot Images for Training Baselines
    # -------------------------------------------------------------------------
    print("\n--- Generating 5 Few-Shot Concept Images from Base Model ---")
    few_shot_prompts = [
        "a painting of starry night by Van Gogh",
        "sunflowers in a vase painted by Van Gogh",
        "a self portrait of Vincent Van Gogh",
        "an olive trees landscape by Van Gogh",
        "a cafe at night in the style of Van Gogh",
    ]
    pipe.unet.load_state_dict(base_unet_state)
    few_shot_images = []
    for idx, prompt in enumerate(few_shot_prompts):
        g = torch.Generator(device).manual_seed(200 + idx)
        with torch.no_grad():
            img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=g).images[0]
        arr = np.array(img).astype(np.float32) / 255.0
        few_shot_images.append(torch.from_numpy(arr).permute(2, 0, 1))
    few_shot_images_tensor = torch.stack(few_shot_images)

    # -------------------------------------------------------------------------
    # Baseline 3: Textual Inversion Positive Control on Base SD 1.4 (1000 steps)
    # -------------------------------------------------------------------------
    print("\n--- Running TI Positive Control on Base SD 1.4 (1000 steps, init='art') ---")
    pipe.unet.load_state_dict(base_unet_state)
    pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
    pipe.tokenizer = copy.deepcopy(clean_tokenizer)

    ti_backend_pos = DiffusersTextualInversionBackend(
        images=few_shot_images_tensor,
        initializer_token="art",
    )
    t_start = time.perf_counter()
    outcome_ti_pos = ti_backend_pos.train_token(
        pipe,
        concept="Van Gogh",
        prompts=few_shot_prompts,
        steps=1000,
        lr=5e-4,
        seed=42,
    )
    ti_pos_time = time.perf_counter() - t_start
    placeholder_pos = outcome_ti_pos.notes.get("placeholder", "<Van Gogh>")
    vg_pattern = re.compile(re.escape("Van Gogh"), re.IGNORECASE)
    ti_prompts_pos = [vg_pattern.sub(placeholder_pos, p) for p in concept_prompts_16]
    c_ti_pos, n_ti_pos, lp_ti_pos, _ = evaluate_condition(
        "TI-PositiveControl-Base", concept_prompts=ti_prompts_pos, base_neutral_tensors=base_neut_imgs
    )
    results["textual_inversion_positive_control"] = {
        "steps": 1000,
        "wall_clock_sec": float(ti_pos_time),
        **paired_bootstrap_metrics(c_ti_pos, n_ti_pos, c_base, n_base, c_esd, n_esd, lp_ti_pos),
    }

    # Reset text encoder & tokenizer
    pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
    pipe.tokenizer = copy.deepcopy(clean_tokenizer)
    del outcome_ti_pos, ti_backend_pos
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Baseline 4: Textual Inversion on Clean ESD-x (1000 steps)
    # -------------------------------------------------------------------------
    print("\n--- Running TI on Clean ESD-x (1000 steps, init='art') ---")
    pipe.unet.load_state_dict(clean_esd_state)
    pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
    pipe.tokenizer = copy.deepcopy(clean_tokenizer)

    ti_backend_esd = DiffusersTextualInversionBackend(
        images=few_shot_images_tensor,
        initializer_token="art",
    )
    t_start = time.perf_counter()
    outcome_ti_esd = ti_backend_esd.train_token(
        pipe,
        concept="Van Gogh",
        prompts=few_shot_prompts,
        steps=1000,
        lr=5e-4,
        seed=42,
    )
    ti_esd_time = time.perf_counter() - t_start
    placeholder_esd = outcome_ti_esd.notes.get("placeholder", "<Van Gogh>")
    ti_prompts_esd = [vg_pattern.sub(placeholder_esd, p) for p in concept_prompts_16]
    c_ti_esd, n_ti_esd, lp_ti_esd, _ = evaluate_condition(
        "TI-1000-CleanESD", concept_prompts=ti_prompts_esd, base_neutral_tensors=base_neut_imgs
    )
    results["textual_inversion_clean_esd"] = {
        "steps": 1000,
        "wall_clock_sec": float(ti_esd_time),
        **paired_bootstrap_metrics(c_ti_esd, n_ti_esd, c_base, n_base, c_esd, n_esd, lp_ti_esd),
    }

    pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
    pipe.tokenizer = copy.deepcopy(clean_tokenizer)
    del outcome_ti_esd, ti_backend_esd
    torch.cuda.empty_cache()
    gc.collect()

    # -------------------------------------------------------------------------
    # Baseline 5: LoRA Convergence Sweep (50, 150, 300, 500, 1000 steps)
    # -------------------------------------------------------------------------
    print("\n--- Running LoRA Convergence Sweep (50, 150, 300, 500, 1000 steps) ---")
    results["lora_sweep"] = {}
    for lora_steps in [50, 150, 300, 500, 1000]:
        print(f"\nTraining LoRA ({lora_steps} steps)...")
        pipe.unet.load_state_dict(clean_esd_state)
        pipe.text_encoder = copy.deepcopy(clean_text_encoder).to(device)
        pipe.tokenizer = copy.deepcopy(clean_tokenizer)

        lora_backend = DiffusersLoRABackend(images=few_shot_images_tensor)
        t_start = time.perf_counter()
        outcome_lora = lora_backend.train(
            pipe,
            concept="Van Gogh",
            prompts=few_shot_prompts,
            steps=lora_steps,
            lr=1e-4,
            rank=4,
            seed=42,
        )
        lora_time = time.perf_counter() - t_start
        c_lora, n_lora, lp_lora, _ = evaluate_condition(
            f"LoRA-{lora_steps}", base_neutral_tensors=base_neut_imgs
        )
        res_lora = paired_bootstrap_metrics(c_lora, n_lora, c_base, n_base, c_esd, n_esd, lp_lora)
        results["lora_sweep"][f"steps_{lora_steps}"] = {
            "steps": lora_steps,
            "wall_clock_sec": float(lora_time),
            "trainable_params": int(outcome_lora.trainable_params),
            **res_lora,
        }
        print(f"LoRA {lora_steps} steps -> Recovery: {res_lora['normalized_recovery']['mean']*100:.2f}% [{res_lora['normalized_recovery']['ci_low']*100:.2f}%, {res_lora['normalized_recovery']['ci_high']*100:.2f}%] | LPIPS: {res_lora['lpips_paired']['mean']:.4f}")

        if hasattr(pipe.unet, "unload"):
            pipe.unet = pipe.unet.unload()
        elif hasattr(pipe.unet, "base_model") and hasattr(pipe.unet.base_model, "model"):
            pipe.unet = pipe.unet.base_model.model
        torch.cuda.empty_cache()
        gc.collect()

    return results


@app.local_entrypoint()
def main():
    print("Launching Baselines Sweep on Modal...")
    res = run_baselines_sweep.remote()
    out_file = Path("baselines_unified_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {out_file}!")


if __name__ == "__main__":
    main()
