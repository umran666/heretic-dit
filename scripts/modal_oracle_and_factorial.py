"""Modal experiment: Oracle-Subspace SVD Ladder and 2x2 Factorial at k=1 with Paired LPIPS.

Protocol:
1. Reuses pinned models from Modal Volume 'heretic-models':
   - Base SD 1.4: CompVis/stable-diffusion-v1-4
   - Clean ESD-x: /root/models/clean_esd_x_sd14_van_gogh.safetensors
   - Official ESD-x: /root/models/official_esd_sd14_van_gogh.pt
   - UCE model: /root/models/uce_sd14_van_gogh.safetensors
2. Unified Evaluation:
   - 16 held-out concept prompts + 16 neutral control prompts.
   - 3 seeds ([42, 123, 999]), total N = 48 concept and N = 48 neutral images per condition.
   - Base SD 1.4 neutral images cached for paired LPIPS and FID calculation.
3. Experiments:
   - Base SD 1.4 reference
   - Clean ESD-x erased baseline
   - Official ESD-x baseline (cross-attn K/V only)
   - UCE baseline
   - Oracle Ladder on Clean ESD-x (k in {1, 2, 4, 8, 16})
   - Oracle Ladder on Official ESD-x (K/V only, k in {1, 2, 4, 8, 16})
   - Oracle Ladder on UCE (k in {1, 2, 4, 8, 16})
   - Per-layer cosine similarity cos(v_text, v1) logged across all 32 cross-attn layers.
   - 2x2 Factorial at k=1:
     * Cell A: (oracle v1, oracle sigma*u)
     * Cell B: (oracle v1, estimated k*)
     * Cell C: (v_text, oracle sigma*u)
     * Cell D: (v_text, estimated k*)
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

app = modal.App("heretic-oracle-and-factorial")

models_volume = modal.Volume.from_name("heretic-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0",
        "diffusers>=0.28.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "safetensors>=0.4.0",
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
    timeout=3600,
)
def run_oracle_and_factorial_sweep() -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    import lpips
    from safetensors.torch import load_file

    from heretic_dit.eval.classifiers import ClipStyleScorer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Oracle & 2x2 Factorial Sweep starting on {torch.cuda.get_device_name(0)} ===")
    models_dir = Path("/root/models")

    # 1. Load Base SD 1.4 Pipeline
    model_id = "CompVis/stable-diffusion-v1-4"
    print(f"Loading Base SD 1.4 from {model_id}...")
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

    # 2. Load Erased Checkpoints from Volume
    clean_esd_file = models_dir / "clean_esd_x_sd14_van_gogh.safetensors"
    official_esd_file = models_dir / "official_esd_sd14_van_gogh.pt"
    uce_file = models_dir / "uce_sd14_van_gogh.safetensors"

    print(f"Loading Clean ESD-x from {clean_esd_file}...")
    clean_esd_state = load_file(str(clean_esd_file))

    print(f"Loading Official ESD-x from {official_esd_file}...")
    official_esd_raw = torch.load(str(official_esd_file), map_location="cpu")
    official_esd_state = official_esd_raw if not isinstance(official_esd_raw, dict) or "state_dict" not in official_esd_raw else official_esd_raw["state_dict"]

    print(f"Loading UCE from {uce_file}...")
    uce_state = load_file(str(uce_file))

    # Cross-attention target matrices
    cross_attn_names = [
        name for name, _ in unet.named_parameters()
        if "attn2.to_k.weight" in name or "attn2.to_v.weight" in name
    ]
    print(f"Identified {len(cross_attn_names)} cross-attention target matrices.")

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

    # -------------------------------------------------------------------------
    # Helper: Generate and evaluate model condition
    # -------------------------------------------------------------------------
    def evaluate_condition(
        state_dict: Dict[str, torch.Tensor],
        tag: str,
        cache_base_neutral: bool = False,
        base_neutral_tensors: List[torch.Tensor] = None,
    ) -> Tuple[List[float], List[float], List[float], List[torch.Tensor]]:
        pipe.unet.load_state_dict(state_dict)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        concept_scores = []
        for prompt in concept_prompts_16:
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

        # Compute paired LPIPS against base reference neutral images
        lpips_scores = []
        if base_neutral_tensors is not None:
            with torch.no_grad():
                for m_img, b_img in zip(neutral_img_tensors, base_neutral_tensors):
                    # Scale to [-1, 1]
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
    c_base, n_base, lp_base, base_neut_imgs = evaluate_condition(
        base_unet_state, "Base SD 1.4", cache_base_neutral=True
    )
    base_metrics = paired_bootstrap_metrics(
        c_base, n_base, c_base, n_base, c_base, n_base, lp_base
    )
    results["base_sd14"] = base_metrics

    # -------------------------------------------------------------------------
    # Baseline 2: Clean ESD-x Control Checkpoint
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Clean ESD-x Baseline ---")
    c_esd, n_esd, lp_esd, _ = evaluate_condition(
        clean_esd_state, "Clean ESD-x", base_neutral_tensors=base_neut_imgs
    )
    clean_esd_metrics = paired_bootstrap_metrics(
        c_esd, n_esd, c_base, n_base, c_esd, n_esd, lp_esd
    )
    results["clean_esd"] = clean_esd_metrics

    # -------------------------------------------------------------------------
    # Baseline 3: Official ESD-x Checkpoint (cross-attn K/V only)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Official ESD-x Checkpoint (cross-attn K/V only) ---")
    official_kv_state = copy.deepcopy(base_unet_state)
    for name in cross_attn_names:
        if name in official_esd_state:
            official_kv_state[name] = official_esd_state[name].cpu()
    c_off, n_off, lp_off, _ = evaluate_condition(
        official_kv_state, "Official ESD-x (K/V)", base_neutral_tensors=base_neut_imgs
    )
    official_esd_metrics = paired_bootstrap_metrics(
        c_off, n_off, c_base, n_base, c_esd, n_esd, lp_off
    )
    results["official_esd_kv"] = official_esd_metrics

    # -------------------------------------------------------------------------
    # Baseline 4: UCE Checkpoint
    # -------------------------------------------------------------------------
    print("\n--- Evaluating UCE Checkpoint ---")
    c_uce, n_uce, lp_uce, _ = evaluate_condition(
        uce_state, "UCE Baseline", base_neutral_tensors=base_neut_imgs
    )
    uce_metrics = paired_bootstrap_metrics(
        c_uce, n_uce, c_base, n_base, c_esd, n_esd, lp_uce
    )
    results["uce_baseline"] = uce_metrics

    # -------------------------------------------------------------------------
    # Experiment A: Oracle SVD Ladder on Clean ESD-x (k in {1, 2, 4, 8, 16})
    # -------------------------------------------------------------------------
    print("\n=== Oracle SVD Ladder on Clean ESD-x ===")
    delta_w_clean: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
    clean_fro_sq = 0.0
    for name in cross_attn_names:
        w_b = base_unet_state[name].float()
        w_e = clean_esd_state[name].float()
        dw = w_b - w_e
        u, s, vh = torch.linalg.svd(dw, full_matrices=False)
        fro_sq = float((dw**2).sum().item())
        clean_fro_sq += fro_sq
        delta_w_clean[name] = (u, s, vh, fro_sq)

    results["oracle_clean_esd"] = {}
    for k in [1, 2, 4, 8, 16]:
        print(f"\nEvaluating Clean ESD-x Oracle Rank k={k}...")
        k_energy = 0.0
        restored = copy.deepcopy(clean_esd_state)
        for name in cross_attn_names:
            u, s, vh, _ = delta_w_clean[name]
            rk = min(k, len(s))
            k_energy += float((s[:rk]**2).sum().item())
            dw_k = (u[:, :rk] * s[:rk]) @ vh[:rk, :]
            restored[name] = clean_esd_state[name] + dw_k.cpu()

        energy_frac = float(k_energy / clean_fro_sq)
        c_k, n_k, lp_k, _ = evaluate_condition(
            restored, f"Oracle-Clean-k{k}", base_neutral_tensors=base_neut_imgs
        )
        res_k = paired_bootstrap_metrics(c_k, n_k, c_base, n_base, c_esd, n_esd, lp_k)
        results["oracle_clean_esd"][f"rank_{k}"] = {
            "rank": k,
            "energy_fraction": energy_frac,
            **res_k,
        }
        print(f"Clean ESD Rank k={k} -> Energy: {energy_frac*100:.1f}% | Recovery: {res_k['normalized_recovery']['mean']*100:.2f}% [{res_k['normalized_recovery']['ci_low']*100:.2f}%, {res_k['normalized_recovery']['ci_high']*100:.2f}%]")

    # -------------------------------------------------------------------------
    # Experiment B: Oracle SVD Ladder on Official ESD-x (K/V only, k in {1, 2, 4, 8, 16})
    # -------------------------------------------------------------------------
    print("\n=== Oracle SVD Ladder on Official ESD-x (K/V only) ===")
    delta_w_off: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
    off_fro_sq = 0.0
    for name in cross_attn_names:
        w_b = base_unet_state[name].float()
        w_e = official_kv_state[name].float()
        dw = w_b - w_e
        u, s, vh = torch.linalg.svd(dw, full_matrices=False)
        fro_sq = float((dw**2).sum().item())
        off_fro_sq += fro_sq
        delta_w_off[name] = (u, s, vh, fro_sq)

    results["oracle_official_esd"] = {}
    for k in [1, 2, 4, 8, 16]:
        print(f"\nEvaluating Official ESD-x Oracle Rank k={k}...")
        k_energy = 0.0
        restored = copy.deepcopy(official_kv_state)
        for name in cross_attn_names:
            u, s, vh, _ = delta_w_off[name]
            rk = min(k, len(s))
            k_energy += float((s[:rk]**2).sum().item())
            dw_k = (u[:, :rk] * s[:rk]) @ vh[:rk, :]
            restored[name] = official_kv_state[name] + dw_k.cpu()

        energy_frac = float(k_energy / off_fro_sq)
        c_k, n_k, lp_k, _ = evaluate_condition(
            restored, f"Oracle-Official-k{k}", base_neutral_tensors=base_neut_imgs
        )
        res_k = paired_bootstrap_metrics(c_k, n_k, c_base, n_base, c_esd, n_esd, lp_k)
        results["oracle_official_esd"][f"rank_{k}"] = {
            "rank": k,
            "energy_fraction": energy_frac,
            **res_k,
        }
        print(f"Official ESD Rank k={k} -> Energy: {energy_frac*100:.1f}% | Recovery: {res_k['normalized_recovery']['mean']*100:.2f}% [{res_k['normalized_recovery']['ci_low']*100:.2f}%, {res_k['normalized_recovery']['ci_high']*100:.2f}%]")

    # -------------------------------------------------------------------------
    # Experiment C: Oracle SVD Ladder on UCE (k in {1, 2, 4, 8, 16})
    # -------------------------------------------------------------------------
    print("\n=== Oracle SVD Ladder on UCE ===")
    delta_w_uce: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}
    uce_fro_sq = 0.0
    for name in cross_attn_names:
        w_b = base_unet_state[name].float()
        w_e = uce_state[name].float()
        dw = w_b - w_e
        u, s, vh = torch.linalg.svd(dw, full_matrices=False)
        fro_sq = float((dw**2).sum().item())
        uce_fro_sq += fro_sq
        delta_w_uce[name] = (u, s, vh, fro_sq)

    results["oracle_uce"] = {}
    for k in [1, 2, 4, 8, 16]:
        print(f"\nEvaluating UCE Oracle Rank k={k}...")
        k_energy = 0.0
        restored = copy.deepcopy(uce_state)
        for name in cross_attn_names:
            u, s, vh, _ = delta_w_uce[name]
            rk = min(k, len(s))
            k_energy += float((s[:rk]**2).sum().item())
            dw_k = (u[:, :rk] * s[:rk]) @ vh[:rk, :]
            restored[name] = uce_state[name] + dw_k.cpu()

        energy_frac = float(k_energy / uce_fro_sq)
        c_k, n_k, lp_k, _ = evaluate_condition(
            restored, f"Oracle-UCE-k{k}", base_neutral_tensors=base_neut_imgs
        )
        res_k = paired_bootstrap_metrics(c_k, n_k, c_base, n_base, c_esd, n_esd, lp_k)
        results["oracle_uce"][f"rank_{k}"] = {
            "rank": k,
            "energy_fraction": energy_frac,
            **res_k,
        }
        print(f"UCE Rank k={k} -> Energy: {energy_frac*100:.1f}% | Recovery: {res_k['normalized_recovery']['mean']*100:.2f}% [{res_k['normalized_recovery']['ci_low']*100:.2f}%, {res_k['normalized_recovery']['ci_high']*100:.2f}%]")

    # -------------------------------------------------------------------------
    # Experiment D: 2x2 Factorial Test at k=1 & Per-Layer Cosine Analysis
    # -------------------------------------------------------------------------
    print("\n=== 2x2 Factorial Experiment at k=1 ===")
    # 1. Compute unsupervised v_text from text encoder
    def encode_text(texts: List[str]) -> torch.Tensor:
        tok = tokenizer(texts, padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
        return text_encoder(tok)[0]  # [B, 77, 768]

    with torch.no_grad():
        emb_pos = encode_text(concept_prompts_16).mean(dim=1)  # [16, 768]
        emb_neut = encode_text(neutral_prompts_16).mean(dim=1)  # [16, 768]
        diff_text = emb_pos.mean(dim=0) - emb_neut.mean(dim=0)
        v_text = (diff_text / torch.linalg.norm(diff_text)).float()  # [768]

        # Related un-erased concept for ROME re-route: Claude Monet / Impressionism
        related_prompts = [
            "a painting in the style of Claude Monet",
            "a post-impressionist landscape painting",
            "an oil painting with expressive brushstrokes by Paul Cezanne",
        ]
        emb_related = encode_text(related_prompts).mean(dim=1).mean(dim=0).float()  # [768]

        # Key covariance on neutral prompts: Cov = (E[x x^T]) + lambda * I
        lambda_reg = 0.05
        cov_neutral = (emb_neut.T @ emb_neut) / emb_neut.shape[0] + lambda_reg * torch.eye(768, device=device)
        cov_neut_inv = torch.linalg.inv(cov_neutral).float()

    # Log per-layer cosine similarity cos(v_text, v1)
    results["layer_cosines"] = {}
    print("\n--- Per-Layer Cosine Similarity cos(v_text, v_1) ---")
    for name in cross_attn_names:
        u, s, vh, _ = delta_w_clean[name]
        v_1 = vh[0, :].to(device).float()
        # Canonicalize sign
        cos_sim = float(torch.abs(torch.dot(v_text, v_1)).item())
        results["layer_cosines"][name] = cos_sim
        print(f"Layer {name[-25:]}: cos(v_text, v_1) = {cos_sim:.4f}")

    mean_cos = float(np.mean(list(results["layer_cosines"].values())))
    print(f"Mean across all 32 cross-attn layers: cos = {mean_cos:.4f}")
    results["mean_layer_cosine"] = mean_cos

    # Define the 4 Factorial Configurations
    # Cell A: Input = Oracle v1, Value = Oracle sigma_1 * u_1
    print("\nEvaluating Factorial Cell A: (Oracle v1, Oracle sigma*u)...")
    state_cell_a = copy.deepcopy(clean_esd_state)
    for name in cross_attn_names:
        u, s, vh, _ = delta_w_clean[name]
        dw_a = (u[:, :1] * s[:1]) @ vh[:1, :]
        state_cell_a[name] = clean_esd_state[name] + dw_a.cpu()
    c_a, n_a, lp_a, _ = evaluate_condition(state_cell_a, "Cell A (Oracle v1, Oracle u)", base_neutral_tensors=base_neut_imgs)
    results["factorial_cell_a"] = paired_bootstrap_metrics(c_a, n_a, c_base, n_base, c_esd, n_esd, lp_a)

    # Cell B: Input = Oracle v1, Value = Estimated ROME re-route k*
    print("\nEvaluating Factorial Cell B: (Oracle v1, Estimated k*)...")
    state_cell_b = copy.deepcopy(clean_esd_state)
    for name in cross_attn_names:
        w_e = clean_esd_state[name].to(device).float()
        u, s, vh, _ = delta_w_clean[name]
        v_1 = vh[0, :].to(device).float()
        # Estimated target value: k* = W_e @ emb_related
        k_star = w_e @ emb_related
        w_v1 = w_e @ v_1
        delta_val = k_star - w_v1
        # Regularized projection: (v1^T Sigma^-1) / (v1^T Sigma^-1 v1)
        inv_v1 = cov_neut_inv @ v_1
        denom = (v_1 @ inv_v1).item()
        dw_b = torch.outer(delta_val, inv_v1) / denom
        state_cell_b[name] = (w_e + dw_b).cpu()
    c_b_eval, n_b, lp_b, _ = evaluate_condition(state_cell_b, "Cell B (Oracle v1, Estimated k*)", base_neutral_tensors=base_neut_imgs)
    results["factorial_cell_b"] = paired_bootstrap_metrics(c_b_eval, n_b, c_base, n_base, c_esd, n_esd, lp_b)

    # Cell C: Input = v_text, Value = Oracle sigma_1 * u_1
    print("\nEvaluating Factorial Cell C: (v_text, Oracle sigma*u)...")
    state_cell_c = copy.deepcopy(clean_esd_state)
    for name in cross_attn_names:
        w_e = clean_esd_state[name].to(device).float()
        u, s, vh, _ = delta_w_clean[name]
        # Match sign of v_text to v_1
        v_1 = vh[0, :].to(device).float()
        sign = 1.0 if torch.dot(v_text, v_1).item() >= 0 else -1.0
        v_in = v_text * sign
        dw_c = torch.outer(u[:, 0].to(device) * s[0].item(), v_in)
        state_cell_c[name] = (w_e + dw_c).cpu()
    c_c, n_c, lp_c, _ = evaluate_condition(state_cell_c, "Cell C (v_text, Oracle u)", base_neutral_tensors=base_neut_imgs)
    results["factorial_cell_c"] = paired_bootstrap_metrics(c_c, n_c, c_base, n_base, c_esd, n_esd, lp_c)

    # Cell D: Input = v_text, Value = Estimated ROME re-route k* (True Unsupervised Method)
    print("\nEvaluating Factorial Cell D: (v_text, Estimated k*)...")
    state_cell_d = copy.deepcopy(clean_esd_state)
    for name in cross_attn_names:
        w_e = clean_esd_state[name].to(device).float()
        k_star = w_e @ emb_related
        w_vtext = w_e @ v_text
        delta_val = k_star - w_vtext
        inv_v = cov_neut_inv @ v_text
        denom = (v_text @ inv_v).item()
        dw_d = torch.outer(delta_val, inv_v) / denom
        state_cell_d[name] = (w_e + dw_d).cpu()
    c_d, n_d, lp_d, _ = evaluate_condition(state_cell_d, "Cell D (v_text, Estimated k*)", base_neutral_tensors=base_neut_imgs)
    results["factorial_cell_d"] = paired_bootstrap_metrics(c_d, n_d, c_base, n_base, c_esd, n_esd, lp_d)

    return results


@app.local_entrypoint()
def main():
    print("Launching Oracle SVD Ladder and 2x2 Factorial on Modal...")
    res = run_oracle_and_factorial_sweep.remote()
    out_file = Path("oracle_and_factorial_results.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {out_file}!")


if __name__ == "__main__":
    main()
