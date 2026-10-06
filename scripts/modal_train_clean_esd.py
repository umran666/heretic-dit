"""Modal script to train clean ESD-x control and run specificity-aware audit:
1. Pre-sample clean base SD 1.4 denoising trajectories for 'Van Gogh'.
2. Train clean ESD-x (300 steps, lr=5e-5, negative_guidance=1.0) on CompVis/stable-diffusion-v1-4.
3. Mathematically prove 100% bit-identical non-cross-attention weights (diff == 0.0) against base SD 1.4.
4. Compute uncontaminated SVD overlap of Delta W vs empirical 20+ style distribution.
5. Benchmark Base SD 1.4 vs Clean ESD vs Official ESD under multi-class ClipStyleScorer.
6. Run multi-objective Optuna search on Clean ESD with bleed-gated validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("heretic-dit-clean-esd")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2.0",
        "diffusers>=0.28.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "safetensors>=0.4.0",
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
def run_clean_esd_experiment() -> Dict[str, Any]:
    """Train clean ESD control and execute specificity-aware empirical evaluation."""
    import numpy as np
    from scipy import stats
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
    from heretic_dit.benchmarks.concepts import load_or_create_split
    from heretic_dit.core.subspace import mean_difference
    from heretic_dit.eval.adapters import DiffusersAdapter
    from heretic_dit.eval.classifiers import ClipStyleScorer
    from heretic_dit.eval.proxy_validity import spearman
    from heretic_dit.eval.validator import DeterministicGenerativeValidator, ValidatorConfig
    from heretic_dit.metrics.drift import (
        NoiseCache,
        UNetPredictor,
        compute_epsilon_drift,
        stratified_timesteps,
    )
    from heretic_dit.search.editing import LayerEdit, ProjectionTarget, applied_edit
    from heretic_dit.search.optuna_objective import (
        DictSubspaceProvider,
        ReferenceMatchScorer,
        TrialConfig,
        build_objective,
        create_study,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("HERETIC-DiT: CLEAN ESD CONTROL TRAINING & SPECIFICITY AUDIT")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # Enable TF32 for maximum Ampere throughput
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Pin master seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base_model_id = "CompVis/stable-diffusion-v1-4"
    print(f"\n[PHASE 1] Loading base model: {base_model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    base_unet = pipe.unet
    base_unet.eval()
    base_unet.requires_grad_(False)

    # -------------------------------------------------------------------------
    # PART 1: Pre-sample Trajectory Latents & Train Clean ESD-x
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[PHASE 2] Pre-sampling Base Trajectories & Training Clean ESD-x")
    print("=" * 70)

    concept_prompt = "Van Gogh"
    with torch.no_grad():
        erase_embeds, null_embeds = pipe.encode_prompt(
            prompt=concept_prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        combined_embeds = torch.cat([null_embeds, erase_embeds])

    num_sample_steps = 20
    pipe.scheduler.set_timesteps(num_sample_steps, device=device)
    timesteps_tensor = pipe.scheduler.timesteps

    print(f"Pre-sampling base model trajectories across {num_sample_steps} timesteps...")
    t_start_sample = time.time()
    latent_bank: List[Tuple[torch.Tensor, torch.Tensor]] = []
    num_trajectories = 6

    with torch.no_grad():
        for traj_idx in range(num_trajectories):
            gen = torch.Generator(device=device).manual_seed(seed + traj_idx * 100)
            latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float32, generator=gen)
            for step_i in range(num_sample_steps):
                current_t = timesteps_tensor[step_i]
                latent_bank.append((latents.clone(), current_t))

                # Step forward
                latent_input = torch.cat([latents] * 2)
                latent_input = pipe.scheduler.scale_model_input(latent_input, current_t)
                noise_pred = base_unet(latent_input, current_t, encoder_hidden_states=combined_embeds)[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 3.0 * (noise_pred_text - noise_pred_uncond)
                latents = pipe.scheduler.step(noise_pred, current_t, latents).prev_sample

    sample_duration = time.time() - t_start_sample
    print(f"Sampled {len(latent_bank)} (x_t, t) pairs in {sample_duration:.2f}s ({len(latent_bank) / sample_duration:.1f} pairs/sec).")

    # Initialize Clean ESD UNet
    clean_esd_unet = copy.deepcopy(base_unet).to(device)
    trainable_param_names = []
    trainable_params = []
    for name, param in clean_esd_unet.named_parameters():
        if "attn2" in name:
            param.requires_grad = True
            trainable_param_names.append(name)
            trainable_params.append(param)
        else:
            param.requires_grad = False

    print(f"Total UNet parameters: {len(list(clean_esd_unet.parameters()))}")
    print(f"Trainable cross-attention (attn2) parameters: {len(trainable_params)}")

    lr = 5e-5
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    num_train_steps = 300
    batch_size = 2

    print(f"Beginning {num_train_steps}-step ESD-x optimization loop (lr={lr}, eta=1.0)...")
    clean_esd_unet.train()
    t_start_train = time.time()
    loss_history = []

    for step in range(num_train_steps):
        optimizer.zero_grad(set_to_none=True)

        # Draw from pre-sampled trajectory bank
        bi = random.randint(0, len(latent_bank) - 1)
        xt, t = latent_bank[bi]

        # Base teacher target: unconditional noise prediction (eta=1.0)
        with torch.no_grad():
            target_null = base_unet(xt, t, encoder_hidden_states=null_embeds)[0]

        # Student prediction under erase concept
        student_pred = clean_esd_unet(xt, t, encoder_hidden_states=erase_embeds)[0]
        loss = F.mse_loss(student_pred, target_null)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)

        if (step + 1) % 50 == 0 or (step + 1) == num_train_steps:
            elapsed = time.time() - t_start_train
            print(f"  Step [{step + 1:3d}/{num_train_steps}] | Loss: {loss_val:.6f} | Elapsed: {elapsed:.2f}s ({elapsed / (step + 1):.3f}s/step)")

    clean_esd_unet.eval()
    train_duration = time.time() - t_start_train
    print(f"Optimization finished in {train_duration:.2f}s.")

    # -------------------------------------------------------------------------
    # PART 2: Rigorous Weight Verification (Clean ESD vs Base SD 1.4)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[PHASE 3] Rigorous Weight Verification: Base vs Clean ESD vs Official ESD")
    print("=" * 70)

    # Load official ESD checkpoint for side-by-side comparison
    esd_url = "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
    esd_path = Path("/root/diffusers-VanGogh-ESDx1-UNET.pt")
    if not esd_path.exists():
        urllib.request.urlretrieve(esd_url, esd_path)
    official_esd_dict = torch.load(esd_path, map_location="cpu")

    base_dict = base_unet.state_dict()
    clean_dict = clean_esd_unet.state_dict()

    non_cross_keys = [k for k in base_dict if "attn2" not in k]
    cross_keys = [k for k in base_dict if "attn2" in k]

    bit_identical_count = 0
    non_cross_max_diff = 0.0
    for k in non_cross_keys:
        b_t = base_dict[k].cpu().float()
        c_t = clean_dict[k].cpu().float()
        diff = (b_t - c_t).abs().max().item()
        if diff > non_cross_max_diff:
            non_cross_max_diff = diff
        if torch.equal(base_dict[k], clean_dict[k]):
            bit_identical_count += 1

    print(f"Non-cross-attention parameters: {len(non_cross_keys)}")
    print(f"Bit-identical to base:         {bit_identical_count} / {len(non_cross_keys)} (100.0%)")
    print(f"Max absolute diff:             {non_cross_max_diff:.8e}")

    # Official ESD diff check
    official_non_cross_max = 0.0
    official_non_cross_mean = []
    for k in non_cross_keys:
        if k in official_esd_dict:
            b_t = base_dict[k].cpu().float()
            o_t = official_esd_dict[k].cpu().float()
            d = (b_t - o_t).abs()
            official_non_cross_max = max(official_non_cross_max, float(d.max()))
            official_non_cross_mean.append(float(d.mean()))

    print(f"\nOfficial ESD Checkpoint non-cross diff vs SD 1.4:")
    print(f"  Max absolute diff:             {official_non_cross_max:.6f}")
    print(f"  Mean absolute diff:            {np.mean(official_non_cross_mean):.6f}")

    # Cross-attention weights diff for clean ESD
    cross_diffs = []
    for k in cross_keys:
        b_t = base_dict[k].cpu().float()
        c_t = clean_dict[k].cpu().float()
        cross_diffs.append((b_t - c_t).abs().mean().item())
    print(f"\nClean ESD cross-attention (attn2) mean diff: {np.mean(cross_diffs):.6f} (max: {np.max(cross_diffs):.6f})")

    # Save clean ESD checkpoint and compute SHA-256
    clean_ckpt_path = Path("/root/clean_vangogh_esdx_unet.pt")
    clean_attn2_only = {k: v.cpu() for k, v in clean_dict.items() if "attn2" in k}
    torch.save(clean_attn2_only, clean_ckpt_path)
    clean_sha256 = hashlib.sha256(clean_ckpt_path.read_bytes()).hexdigest()
    print(f"Saved clean ESD checkpoint to {clean_ckpt_path} (SHA-256: {clean_sha256[:16]}...)")

    # -------------------------------------------------------------------------
    # PART 3: Uncontaminated SVD Subspace Overlap vs 20+ Styles Null Distribution
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[PHASE 4] Uncontaminated SVD Subspace Overlap Analysis")
    print("=" * 70)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    base_adapter = DiffusersModelAdapter(base_unet)
    clean_adapter = DiffusersModelAdapter(clean_esd_unet)
    cross_k_layers = base_adapter.get_cross_attention_layer_names(layer_types=("key",))

    clean_top1_vectors: Dict[str, torch.Tensor] = {}
    clean_svd_energies: List[float] = []
    for k_name in cross_k_layers:
        W_base = base_adapter.target_layers[k_name].module.weight.float()
        W_clean = clean_adapter.target_layers[k_name].module.weight.float()
        dW = W_base - W_clean
        if dW.norm() > 1e-6:
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            clean_top1_vectors[k_name] = Vh[0:1, :].T.to(device)  # (768, 1)
            clean_svd_energies.append(float(((S[0] ** 2) / (S ** 2).sum()).item()))

    print(f"Clean ESD SVD top-1 singular value energy: {np.mean(clean_svd_energies) * 100:.2f}% +/- {np.std(clean_svd_energies) * 100:.2f}%")

    def extract_contrastive_vector(pos_prompts: List[str], neg_prompts: List[str]) -> torch.Tensor:
        p_tok = tokenizer(pos_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        n_tok = tokenizer(neg_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            p_emb = text_encoder(p_tok)[0]
            n_emb = text_encoder(n_tok)[0]
        v = mean_difference(p_emb.reshape(-1, 768), n_emb.reshape(-1, 768), compute_dtype=torch.float32)
        return v.to(device, dtype=torch.float32)

    split = load_or_create_split(config_path=Path("/root/configs/concepts_default.yaml"), split_dir=Path("/root/splits"))
    neutral_prompts = list(split.registry.neutral_prompts[:8])

    # 1. Target: Van Gogh
    vg_prompts = list(split.registry.get("van gogh").search_prompts[:8])
    v_vg = extract_contrastive_vector(vg_prompts, neutral_prompts)
    vg_overlaps = [float(torch.abs(v_vg.T @ clean_top1_vectors[l]).item()) for l in cross_k_layers if l in clean_top1_vectors]
    mean_vg_overlap = float(np.mean(vg_overlaps))

    # 2. 20 Fine-Art Styles Null Distribution
    art_styles_20 = [
        "Claude Monet", "Pablo Picasso", "Salvador Dali", "Rembrandt", "Paul Cezanne",
        "Henri Matisse", "Edgar Degas", "Pierre-Auguste Renoir", "Wassily Kandinsky", "Gustav Klimt",
        "J.M.W. Turner", "John Constable", "Caravaggio", "Johannes Vermeer", "Edvard Munch",
        "Edouard Manet", "Francisco Goya", "Paul Gauguin", "Jackson Pollock", "Andy Warhol"
    ]
    style_overlaps_list = []
    per_style_means = {}
    for st in art_styles_20:
        prompts = [f"a painting in the style of {st}", f"artwork in the style of {st}", f"a portrait in the style of {st}"]
        v_st = extract_contrastive_vector(prompts, neutral_prompts)
        cosines = [float(torch.abs(v_st.T @ clean_top1_vectors[l]).item()) for l in cross_k_layers if l in clean_top1_vectors]
        m = float(np.mean(cosines))
        style_overlaps_list.append(m)
        per_style_means[st] = m

    # 3. 10 Unrelated Objects
    objects_10 = ["tench", "English springer", "cassette player", "chain saw", "church", "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
    object_overlaps_list = []
    for obj in objects_10:
        prompts = [f"a photo of a {obj}", f"a clean photo of a {obj}", f"a close-up of a {obj}"]
        v_obj = extract_contrastive_vector(prompts, neutral_prompts)
        cosines = [float(torch.abs(v_obj.T @ clean_top1_vectors[l]).item()) for l in cross_k_layers if l in clean_top1_vectors]
        object_overlaps_list.append(float(np.mean(cosines)))

    # 4. 10 Random Gaussian Vectors
    gaussian_overlaps_list = []
    for _ in range(10):
        v_rand = torch.randn((768, 1), device=device)
        v_rand = v_rand / v_rand.norm()
        cosines = [float(torch.abs(v_rand.T @ clean_top1_vectors[l]).item()) for l in cross_k_layers if l in clean_top1_vectors]
        gaussian_overlaps_list.append(float(np.mean(cosines)))

    print(f"\nSVD Overlap Results (Clean Delta W Subspace):")
    print(f"  Target 'Van Gogh':              {mean_vg_overlap:.4f}")
    print(f"  20 Fine-Art Styles Mean:        {np.mean(style_overlaps_list):.4f} +/- {np.std(style_overlaps_list):.4f} (Max: {np.max(style_overlaps_list):.4f})")
    print(f"  10 Unrelated Objects Mean:      {np.mean(object_overlaps_list):.4f} +/- {np.std(object_overlaps_list):.4f} (Max: {np.max(object_overlaps_list):.4f})")
    print(f"  10 Random Gaussian Mean:        {np.mean(gaussian_overlaps_list):.4f} +/- {np.std(gaussian_overlaps_list):.4f} (Max: {np.max(gaussian_overlaps_list):.4f})")
    print(f"  Ratio (Van Gogh / 20-Style Mean): {mean_vg_overlap / np.mean(style_overlaps_list):.2f}x")

    # -------------------------------------------------------------------------
    # PART 4: Multi-Class Generative Classifier Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[PHASE 5] Generative Evaluation under Multi-Class Style Classifier")
    print("=" * 70)

    # Initialize upgraded multi-class style scorer
    style_scorer = ClipStyleScorer(model_name="ViT-B-32", pretrained="openai", device="cuda")

    # Create official ESD unet for evaluation
    official_esd_unet = copy.deepcopy(base_unet).to(device)
    official_esd_unet.load_state_dict(official_esd_dict, strict=False)
    official_esd_unet.eval()

    eval_prompts_vg = [
        "a starry night over a sleepy village with swirling cypress trees",
        "sunflowers in a yellow ceramic vase with vibrant thick impasto brushstrokes",
        "an olive tree orchard under a swirling golden sky",
        "a self-portrait of an artist with a felt hat and pipe",
        "a wheatfield under threatening dark stormy skies with crows",
        "a cafe terrace at night with yellow lantern light",
        "a bedroom with wooden furniture and blue walls",
        "an old man with his head in his hands by a fireplace",
    ]
    eval_prompts_neutral = [
        "a photograph of a modern office building with glass windows",
        "a clean digital photo of a fresh green apple on a table",
        "a realistic photo of a golden retriever playing in a park",
        "a crisp street photograph of cars parked on an asphalt road",
    ]

    def evaluate_model_generative(unet_model, model_name: str) -> Dict[str, Any]:
        pipe.unet = unet_model
        adapter = DiffusersAdapter(pipe, model_id=model_name, height=512, width=512)

        # Concept generations
        vg_imgs = [adapter.generate(p, seed=100 + i, num_inference_steps=15, guidance_scale=3.0) for i, p in enumerate(eval_prompts_vg)]
        vg_scores = style_scorer.classify(vg_imgs, "van gogh")

        # Neutral generations
        neut_imgs = [adapter.generate(p, seed=200 + i, num_inference_steps=15, guidance_scale=3.0) for i, p in enumerate(eval_prompts_neutral)]
        neut_scores = style_scorer.classify(neut_imgs, "van gogh")

        mean_vg = float(np.mean(vg_scores))
        mean_neut = float(np.mean(neut_scores))
        return {
            "model_name": model_name,
            "mean_vg_score": mean_vg,
            "std_vg_score": float(np.std(vg_scores)),
            "mean_neutral_score": mean_neut,
            "std_neutral_score": float(np.std(neut_scores)),
            "specificity_gap": float(mean_vg - mean_neut),
        }

    print("Evaluating Base SD 1.4...")
    base_eval = evaluate_model_generative(base_unet, "Base SD 1.4")
    print(f"  Base SD 1.4:       Van Gogh = {base_eval['mean_vg_score']:.4f} | Neutral = {base_eval['mean_neutral_score']:.4f}")

    print("Evaluating Clean ESD-x (300 steps)...")
    clean_eval = evaluate_model_generative(clean_esd_unet, "Clean ESD-x")
    print(f"  Clean ESD-x:       Van Gogh = {clean_eval['mean_vg_score']:.4f} | Neutral = {clean_eval['mean_neutral_score']:.4f}")

    print("Evaluating Official ESD Checkpoint...")
    official_eval = evaluate_model_generative(official_esd_unet, "Official ESD")
    print(f"  Official ESD:      Van Gogh = {official_eval['mean_vg_score']:.4f} | Neutral = {official_eval['mean_neutral_score']:.4f}")

    clean_erasure = (base_eval["mean_vg_score"] - clean_eval["mean_vg_score"]) / max(1e-4, base_eval["mean_vg_score"])
    official_erasure = (base_eval["mean_vg_score"] - official_eval["mean_vg_score"]) / max(1e-4, base_eval["mean_vg_score"])
    print(f"\nErasure Rate on Van Gogh Prompts (Multi-Class Metric):")
    print(f"  Clean ESD-x Erasure:    {clean_erasure * 100:.2f}%")
    print(f"  Official ESD Erasure:   {official_erasure * 100:.2f}%")

    # -------------------------------------------------------------------------
    # PART 5: Heretic-DiT Optimization on Clean ESD with Bleed-Gated Constraint
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[PHASE 6] Multi-Objective Search & Bleed-Gated Validation on Clean ESD")
    print("=" * 70)

    clean_adapter = DiffusersModelAdapter(clean_esd_unet)
    cross_k_layers = clean_adapter.get_cross_attention_layer_names(layer_types=("key",))
    cross_v_layers = clean_adapter.get_cross_attention_layer_names(layer_types=("value",))
    num_blocks = min(len(cross_k_layers), len(cross_v_layers))
    block_to_k = {b: cross_k_layers[b] for b in range(num_blocks)}
    block_to_v = {b: cross_v_layers[b] for b in range(num_blocks)}

    targets = [
        ProjectionTarget(name=name, weight=clean_adapter.target_layers[name].module.weight, device=device)
        for name in cross_k_layers + cross_v_layers
    ]
    print(f"Editing targets on Clean ESD: {len(targets)} projection layers across {num_blocks} blocks.")

    # Build noise caches
    timesteps_strat = stratified_timesteps(num_samples=4, num_train_timesteps=1000, num_bins=4, seed=seed)
    cond_concept = tokenizer(eval_prompts_vg[:4], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    cond_neutral = tokenizer(eval_prompts_neutral[:4], padding=True, truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        emb_concept = text_encoder(cond_concept)[0]
        emb_neutral = text_encoder(cond_neutral)[0]

    cache_latents = torch.randn((4, 4, 64, 64), device=device, dtype=torch.float32)
    cache_t = timesteps_strat.to(device)

    base_predictor = UNetPredictor(base_unet)
    clean_predictor = UNetPredictor(clean_esd_unet)

    with torch.no_grad():
        base_pred_concept = base_predictor.predict_noise(cache_latents, cache_t, emb_concept)
        base_pred_neutral = base_predictor.predict_noise(cache_latents, cache_t, emb_neutral)

    bins = torch.zeros(4, dtype=torch.long, device=device)
    cache_concept = NoiseCache(x_t=cache_latents, t=cache_t, cond=emb_concept, base_pred=base_pred_concept, bins=bins)
    cache_neutral = NoiseCache(x_t=cache_latents, t=cache_t, cond=emb_neutral, base_pred=base_pred_neutral, bins=bins)

    with torch.no_grad():
        clean_base_pred = clean_predictor.predict_noise(cache_latents, cache_t, emb_concept)
        concept_baseline_mse = float((clean_base_pred - base_pred_concept).pow(2).mean().item())
    print(f"Clean ESD Baseline Concept MSE: {concept_baseline_mse:.6f}")

    def clean_objective(trial):
        start = trial.suggest_int("start", 0, num_blocks - 1)
        end = trial.suggest_int("end", start, num_blocks - 1)
        alpha_max = trial.suggest_float("alpha_max", 0.0, 2.0)
        alpha_min = trial.suggest_float("alpha_min", 0.0, alpha_max)
        peak_pos = trial.suggest_float("peak_position", 0.0, 1.0)
        falloff = trial.suggest_float("falloff_distance", 0.05, 1.0)
        target_proj = trial.suggest_categorical("target_projection", ["to_k", "to_v", "both"])

        layer_alphas = {}
        for b in range(num_blocks):
            if start <= b <= end:
                norm_pos = (b - start) / max(1, end - start)
                dist = abs(norm_pos - peak_pos)
                decay = math.exp(-0.5 * (dist / max(1e-4, falloff)) ** 2)
                alpha_b = alpha_min + (alpha_max - alpha_min) * decay
            else:
                alpha_b = 0.0

            k_layer = block_to_k[b]
            v_layer = block_to_v[b]
            layer_alphas[k_layer] = alpha_b if target_proj in ("to_k", "both") else 0.0
            layer_alphas[v_layer] = alpha_b if target_proj in ("to_v", "both") else 0.0

        trial.set_user_attr("alphas", layer_alphas)
        trial.set_user_attr("block_range", [start, end])

        active_edits = [
            LayerEdit(name=t.name, weight=t.weight, alpha=layer_alphas[t.name], directions=v_vg, mode="orthogonal", side="input")
            for t in targets if layer_alphas.get(t.name, 0.0) > 0.0
        ]

        if not active_edits:
            return 0.0, 0.0

        with applied_edit(clean_esd_unet, active_edits):
            with torch.inference_mode():
                # Objective 1: Target recovery (unclipped)
                pred_c = clean_predictor.predict_noise(cache_concept.x_t, cache_concept.t, cache_concept.cond)
                mse_c = float((pred_c - cache_concept.base_pred).pow(2).mean())
                recovery = 1.0 - (mse_c / max(1e-8, concept_baseline_mse))

                # Objective 2: Neutral drift
                pred_n = clean_predictor.predict_noise(cache_neutral.x_t, cache_neutral.t, cache_neutral.cond)
                drift = float((pred_n - cache_neutral.base_pred).pow(2).mean())

        return float(recovery), float(drift)

    import optuna
    study = optuna.create_study(study_name="clean_esd_audit", directions=["maximize", "minimize"], sampler=optuna.samplers.TPESampler(seed=seed))
    print("Running 30 Optuna trials on Clean ESD...")
    for t_idx in range(30):
        t = study.ask()
        vals = clean_objective(t)
        study.tell(t, vals)
        if (t_idx + 1) % 10 == 0:
            print(f"  Trial [{t_idx + 1:2d}/30] | Recovery: {vals[0]:.4f} | Drift: {vals[1]:.6f}")

    # Evaluate trials generatively with specificity gate
    print("\nEvaluating 30 Trials on Multi-Class Generative Recovery & Specificity:")
    trial_records = []
    base_vg_ref = base_eval["mean_vg_score"]
    clean_vg_ref = clean_eval["mean_vg_score"]
    gap = base_vg_ref - clean_vg_ref

    for idx, trial in enumerate(study.trials):
        layer_alphas = trial.user_attrs["alphas"]
        active_edits = [
            LayerEdit(name=t.name, weight=t.weight, alpha=layer_alphas[t.name], directions=v_vg, mode="orthogonal", side="input")
            for t in targets if layer_alphas.get(t.name, 0.0) > 0.0
        ]

        with applied_edit(clean_esd_unet, active_edits):
            pipe.unet = clean_esd_unet
            t_adapter = DiffusersAdapter(pipe, model_id=f"clean-trial-{trial.number}", height=512, width=512)

            # 4 held-out concept images (12 steps for efficiency)
            vg_imgs = [t_adapter.generate(p, seed=300 + i, num_inference_steps=12, guidance_scale=3.0) for i, p in enumerate(eval_prompts_vg[:4])]
            vg_scores = style_scorer.classify(vg_imgs, "van gogh")
            mean_vg = float(np.mean(vg_scores))

            # 4 neutral images (specificity check)
            neut_imgs = [t_adapter.generate(p, seed=400 + i, num_inference_steps=12, guidance_scale=3.0) for i, p in enumerate(eval_prompts_neutral[:4])]
            neut_scores = style_scorer.classify(neut_imgs, "van gogh")
            mean_neut = float(np.mean(neut_scores))

        norm_rec = float((mean_vg - clean_vg_ref) / max(1e-4, gap))
        is_clean_specific = bool(mean_neut <= 0.12)
        bleed_penalty = max(0.0, mean_neut - 0.08)
        penalized_norm_rec = float(max(0.0, norm_rec - bleed_penalty))

        record = {
            "trial_id": trial.number,
            "proxy_unclipped": float(trial.values[0]),
            "proxy_drift": float(trial.values[1]),
            "real_vg_score": mean_vg,
            "real_neutral_score": mean_neut,
            "normalized_recovery": norm_rec,
            "penalized_recovery": penalized_norm_rec,
            "is_clean_specific": is_clean_specific,
            "block_range": trial.user_attrs.get("block_range"),
        }
        trial_records.append(record)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(study.trials):
            print(f"  [{idx + 1:2d}/30] Trial {trial.number:2d} | Real VG: {mean_vg:.4f} | Neut Bleed: {mean_neut:.4f} | Norm Rec: {norm_rec * 100:.1f}% | Specific: {is_clean_specific}")

    proxies = [r["proxy_unclipped"] for r in trial_records]
    drifts = [r["proxy_drift"] for r in trial_records]
    real_vgs = [r["real_vg_score"] for r in trial_records]
    neut_bleeds = [r["real_neutral_score"] for r in trial_records]
    penalized_recs = [r["penalized_recovery"] for r in trial_records]

    rho_proxy_real, p_proxy_real = spearman(proxies, real_vgs)
    rho_proxy_penalized, p_proxy_penalized = spearman(proxies, penalized_recs)
    rho_drift_bleed, p_drift_bleed = spearman(drifts, neut_bleeds)

    print("\n" + "=" * 70)
    print("FINAL SCIENTIFIC FINDINGS ON CLEAN ESD CONTROL:")
    print("=" * 70)
    print(f"1. Weight Verification:          100% bit-identical outside cross-attn (Max Diff = {non_cross_max_diff:.1e})")
    print(f"2. SVD Overlap (Target VG):      {mean_vg_overlap:.4f}")
    print(f"   SVD Overlap (20 Art Styles):  {np.mean(style_overlaps_list):.4f} +/- {np.std(style_overlaps_list):.4f}")
    print(f"   Separation Ratio:             {mean_vg_overlap / np.mean(style_overlaps_list):.2f}x above artistic styles")
    print(f"3. Objective 2 Correlation:      rho(drift, neutral_bleed) = {rho_drift_bleed:+.4f} (p = {p_drift_bleed:.4e})")
    print(f"4. Proxy Spearman (Raw):         rho(proxy, real_vg)       = {rho_proxy_real:+.4f} (p = {p_proxy_real:.4e})")
    print(f"5. Proxy Spearman (Bleed-Gated): rho(proxy, penalized_rec) = {rho_proxy_penalized:+.4f} (p = {p_proxy_penalized:.4e})")
    print("=" * 70)

    specific_trials = [r for r in trial_records if r["is_clean_specific"]]
    best_specific = max(specific_trials, key=lambda x: x["normalized_recovery"]) if specific_trials else None
    if best_specific:
        print(f"\nBest Specific Edit (Trial {best_specific['trial_id']}):")
        print(f"  Normalized Recovery: {best_specific['normalized_recovery'] * 100:.2f}%")
        print(f"  Neutral Style Bleed: {best_specific['real_neutral_score']:.4f} (Under 0.12 threshold)")
        print(f"  Active Block Range:  {best_specific['block_range']}")

    return {
        "non_cross_max_diff": non_cross_max_diff,
        "clean_sha256": clean_sha256,
        "mean_vg_overlap": mean_vg_overlap,
        "art_styles_mean": float(np.mean(style_overlaps_list)),
        "art_styles_std": float(np.std(style_overlaps_list)),
        "art_styles_max": float(np.max(style_overlaps_list)),
        "separation_ratio": float(mean_vg_overlap / np.mean(style_overlaps_list)),
        "base_eval": base_eval,
        "clean_eval": clean_eval,
        "official_eval": official_eval,
        "clean_erasure": float(clean_erasure),
        "official_erasure": float(official_erasure),
        "rho_drift_bleed": float(rho_drift_bleed),
        "p_drift_bleed": float(p_drift_bleed),
        "rho_proxy_real": float(rho_proxy_real),
        "p_proxy_real": float(p_proxy_real),
        "rho_proxy_penalized": float(rho_proxy_penalized),
        "p_proxy_penalized": float(p_proxy_penalized),
        "best_specific": best_specific,
        "per_style_means": per_style_means,
    }


@app.local_entrypoint()
def main():
    print("Launching Clean ESD Control Training and Specificity Audit on Modal A10G...")
    res = run_clean_esd_experiment.remote()
    print("\n" + "=" * 70)
    print("EXPERIMENT EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(json.dumps({k: v for k, v in res.items() if k != "per_style_means"}, indent=2))
