"""Modal diagnostic script to address scientific critique:
1. Tensor-by-tensor diff of ESD vs SD 1.4 in full FP32 precision.
2. Text-anisotropy null distribution for SVD overlap across 10 objects + 5 styles.
3. Specificity audit: neutral prompt style score and drift for top vs bottom edits.
4. Proxy v1 (raw epsilon) vs Proxy v2 (guidance delta epsilon(c) - epsilon(empty)) vs unclipped vs timestep window.
5. Test-retest classifier reliability ceiling on generative validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal

app = modal.App("heretic-dit-diagnostic-audit")

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
    gpu="T4",
    timeout=1800,
)
def run_diagnostics() -> Dict[str, Any]:
    """Execute all five diagnostic checks on Tesla T4 GPU."""
    import numpy as np
    from scipy import stats
    import torch
    import torch.nn.functional as F
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
    from heretic_dit.benchmarks.concepts import load_or_create_split
    from heretic_dit.core.subspace import mean_difference
    from heretic_dit.eval.adapters import DiffusersAdapter
    from heretic_dit.eval.classifiers import ClipStyleScorer
    from heretic_dit.eval.proxy_validity import ProxyTrial, evaluate_proxy_validity, spearman
    from heretic_dit.eval.validator import DeterministicGenerativeValidator, ValidatorConfig
    from heretic_dit.interfaces import EditSpec
    from heretic_dit.metrics.drift import (
        NoiseCache,
        UNetPredictor,
        add_noise,
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
    print("HERETIC-DiT: RIGOROUS EMPIRICAL DIAGNOSTICS & AUDIT")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    # Fetch official ESD checkpoint
    esd_url = "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
    esd_path = Path("/root/diffusers-VanGogh-ESDx1-UNET.pt")
    if not esd_path.exists():
        print(f"Downloading official ESD checkpoint from {esd_url}...")
        urllib.request.urlretrieve(esd_url, esd_path)

    esd_state_dict = torch.load(esd_path, map_location="cpu")
    print(f"Loaded ESD checkpoint: {len(esd_state_dict)} parameter keys.")

    # -------------------------------------------------------------------------
    # CHECK 1: Tensor-by-Tensor Diff of ESD vs SD 1.4 (Full Float32 Precision)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[DIAGNOSTIC 1] Full FP32 Tensor-by-Tensor Diff: ESD vs SD 1.4")
    print("=" * 70)

    unet_sd14_fp32 = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float32
    )
    sd14_dict_fp32 = unet_sd14_fp32.state_dict()

    diff_records = []
    category_diffs: Dict[str, List[float]] = {
        "cross_attn_k_v": [],
        "cross_attn_q_out": [],
        "self_attn": [],
        "resnets_convs": [],
        "time_embed": [],
        "other": [],
    }

    for k, esd_tensor in esd_state_dict.items():
        if k not in sd14_dict_fp32:
            continue
        base_tensor = sd14_dict_fp32[k]
        diff = (base_tensor.float() - esd_tensor.float()).abs()
        max_d = float(diff.max())
        mean_d = float(diff.mean())

        # Categorize
        if "attn2.to_k" in k or "attn2.to_v" in k:
            cat = "cross_attn_k_v"
        elif "attn2.to_q" in k or "attn2.to_out" in k:
            cat = "cross_attn_q_out"
        elif "attn1" in k:
            cat = "self_attn"
        elif "conv" in k or "resnets" in k or "norm" in k:
            cat = "resnets_convs"
        elif "time_embed" in k:
            cat = "time_embed"
        else:
            cat = "other"

        category_diffs[cat].append(max_d)
        if max_d > 0.0:
            diff_records.append({
                "key": k,
                "category": cat,
                "max_diff": max_d,
                "mean_diff": mean_d,
                "esd_dtype": str(esd_tensor.dtype),
                "sd14_dtype": str(base_tensor.dtype),
            })

    print("\nSummary by Architectural Category (ESD vs SD 1.4 FP32):")
    for cat, diffs in category_diffs.items():
        if not diffs:
            continue
        non_zero = sum(d > 1e-7 for d in diffs)
        max_val = max(diffs) if diffs else 0.0
        mean_val = float(np.mean(diffs)) if diffs else 0.0
        print(f"  - {cat:20s}: {len(diffs):3d} tensors | {non_zero:3d} modified | max diff = {max_val:.8f} | mean = {mean_val:.8f}")

    # Sort non-cross-attn diffs to see largest outliers
    non_cross_diffs = [r for r in diff_records if r["category"] != "cross_attn_k_v"]
    non_cross_diffs.sort(key=lambda x: x["max_diff"], reverse=True)
    print(f"\nTop 10 Differing Tensors Outside Cross-Attn K/V:")
    for r in non_cross_diffs[:10]:
        print(f"  {r['key'][:55]:55s} | {r['category']:16s} | max={r['max_diff']:.6f} | mean={r['mean_diff']:.8f} | dtype={r['esd_dtype']}")

    del unet_sd14_fp32, sd14_dict_fp32
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # CHECK 2: Empirical Text-Anisotropy Null Distribution for SVD Overlap
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[DIAGNOSTIC 2] Text-Anisotropy Null Distribution for SVD Overlap")
    print("=" * 70)

    # Load SD 1.4 pipeline components in fp16
    base_model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    base_unet = UNet2DConditionModel.from_pretrained(
        base_model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    base_unet.eval()
    text_encoder.eval()

    erased_unet = copy.deepcopy(base_unet)
    erased_unet.load_state_dict(esd_state_dict, strict=False)
    erased_unet.eval().to(device)

    base_adapter = DiffusersModelAdapter(base_unet)
    erased_adapter = DiffusersModelAdapter(erased_unet)
    cross_k_layers = base_adapter.get_cross_attention_layer_names(layer_types=("key",))
    cross_v_layers = base_adapter.get_cross_attention_layer_names(layer_types=("value",))

    # Compute ESD delta W SVD top singular vectors
    esd_top1_vectors: Dict[str, torch.Tensor] = {}
    svd_energies: List[float] = []
    for k_name in cross_k_layers:
        W_base = base_adapter.target_layers[k_name].module.weight.float()
        W_esd = erased_adapter.target_layers[k_name].module.weight.float()
        dW = W_base - W_esd
        if dW.norm() > 1e-6:
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            esd_top1_vectors[k_name] = Vh[0:1, :].T.to(device)  # (768, 1)
            svd_energies.append(float(((S.detach()[0] ** 2) / (S.detach() ** 2).sum()).item()))

    # Build contrastive direction helper
    def extract_contrastive_vector(pos_prompts: List[str], neg_prompts: List[str]) -> torch.Tensor:
        p_tok = tokenizer(pos_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        n_tok = tokenizer(neg_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            p_emb = text_encoder(p_tok)[0]
            n_emb = text_encoder(n_tok)[0]
        v = mean_difference(p_emb.reshape(-1, 768), n_emb.reshape(-1, 768), compute_dtype=torch.float32)
        return v.to(device, dtype=torch.float32)  # (768, 1)

    split = load_or_create_split(config_path=Path("/root/configs/concepts_default.yaml"), split_dir=Path("/root/splits"))
    neutral_prompts = list(split.registry.neutral_prompts[:8])

    # 1. Target: Van Gogh
    vg_prompts = list(split.registry.get("van gogh").search_prompts[:8])
    v_vangogh = extract_contrastive_vector(vg_prompts, neutral_prompts)

    # 2. Ten Unrelated Objects (Imagenette)
    unrelated_objects = [
        "tench", "English springer", "cassette player", "chain saw", "church",
        "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
    ]
    object_overlaps = []
    for obj in unrelated_objects:
        prompts = [f"a photo of a {obj}", f"a clean photo of a {obj}", f"a close-up of a {obj}"]
        v_obj = extract_contrastive_vector(prompts, neutral_prompts)
        layer_cosines = [float(torch.abs(v_obj.T @ esd_top1_vectors[l]).item()) for l in cross_k_layers if l in esd_top1_vectors]
        object_overlaps.append(float(np.mean(layer_cosines)))

    # 3. Five Other Art Styles
    other_styles = ["Claude Monet", "Pablo Picasso", "Salvador Dali", "Rembrandt", "watercolor painting"]
    style_overlaps = []
    for st in other_styles:
        prompts = [f"a painting in the style of {st}", f"artwork in the style of {st}", f"a portrait in the style of {st}"]
        v_st = extract_contrastive_vector(prompts, neutral_prompts)
        layer_cosines = [float(torch.abs(v_st.T @ esd_top1_vectors[l]).item()) for l in cross_k_layers if l in esd_top1_vectors]
        style_overlaps.append(float(np.mean(layer_cosines)))

    # 4. Pure Gaussian Random Vectors
    gen = torch.Generator(device=device).manual_seed(9999)
    random_overlaps = []
    for _ in range(10):
        rand_v = torch.randn(768, 1, generator=gen, device=device)
        rand_v = rand_v / rand_v.norm()
        layer_cosines = [float(torch.abs(rand_v.T @ esd_top1_vectors[l]).item()) for l in cross_k_layers if l in esd_top1_vectors]
        random_overlaps.append(float(np.mean(layer_cosines)))

    # Target Van Gogh overlap
    vg_layer_cosines = [float(torch.abs(v_vangogh.T @ esd_top1_vectors[l]).item()) for l in cross_k_layers if l in esd_top1_vectors]
    vg_overlap = float(np.mean(vg_layer_cosines))

    print(f"\nSVD Overlap Comparison Across Null Distributions:")
    print(f"  Gaussian Random Vector Null (N=10):    Mean = {np.mean(random_overlaps):.4f} +/- {np.std(random_overlaps):.4f} (Max: {np.max(random_overlaps):.4f})")
    print(f"  Unrelated Objects (Imagenette, N=10):  Mean = {np.mean(object_overlaps):.4f} +/- {np.std(object_overlaps):.4f} (Max: {np.max(object_overlaps):.4f})")
    print(f"  Other Art Styles (N=5):                Mean = {np.mean(style_overlaps):.4f} +/- {np.std(style_overlaps):.4f} (Max: {np.max(style_overlaps):.4f})")
    print(f"  Target 'Van Gogh' Overlap:             {vg_overlap:.4f}")

    all_unrelated_text = object_overlaps + style_overlaps
    z_score_text = (vg_overlap - np.mean(all_unrelated_text)) / np.std(all_unrelated_text)
    percentile_text = stats.percentileofscore(all_unrelated_text, vg_overlap)
    print(f"  Van Gogh vs Empirical Text Null:       z = +{z_score_text:.2f} sigma (Percentile: {percentile_text:.1f}%)")

    # -------------------------------------------------------------------------
    # CHECK 3 & 4: Dev-Set Proxy Comparison, Specificity, & Generative Power
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[DIAGNOSTIC 3 & 4] Proxy v1 vs Guidance Delta v2, Specificity & Ceiling")
    print("=" * 70)

    # Set up predictors and caches
    base_predictor = UNetPredictor(model=base_unet, prediction_type="epsilon")
    erased_predictor = UNetPredictor(model=erased_unet, prediction_type="epsilon")

    concept_entry = split.registry.get("van gogh")
    concept_eval_prompts = list(concept_entry.search_prompts[:4])
    heldout_prompts = list(concept_entry.heldout_prompts[:4])
    neutral_eval_prompts = list(split.registry.neutral_prompts[:4])

    del esd_state_dict
    torch.cuda.empty_cache()

    # Tokenize prompts
    c_tokens = tokenizer(concept_eval_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    n_tokens = tokenizer(neutral_eval_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    uncond_tokens = tokenizer([""] * len(concept_eval_prompts), padding=True, truncation=True, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        c_cond = text_encoder(c_tokens)[0]
        n_cond = text_encoder(n_tokens)[0]
        uncond_cond = text_encoder(uncond_tokens)[0]

    # Caches:
    # 1. Standard stratified cache (uniform bins across [0, 1000))
    concept_latents = torch.randn(len(concept_eval_prompts), 4, 64, 64, device=device, dtype=torch.float16)
    cache_standard = NoiseCache.build(base_predictor=base_predictor, latents=concept_latents, timesteps=None, cond=c_cond, seed=42)

    # 2. Windowed cache restricted to semantic synthesis window t in [100, 500]
    t_window = torch.randint(100, 500, (len(concept_eval_prompts),), device="cpu")
    cache_windowed = NoiseCache.build(base_predictor=base_predictor, latents=concept_latents, timesteps=t_window, cond=c_cond, seed=42)

    # 3. Guidance Delta cache: compute ref_guidance_delta = base(c) - base(empty)
    with torch.inference_mode():
        ref_c_pred = cache_standard.base_pred
        ref_uncond_pred = base_predictor.predict_noise(cache_standard.x_t, cache_standard.t, uncond_cond)
        ref_guidance_delta = (ref_c_pred - ref_uncond_pred).detach()

        erased_c_pred = erased_predictor.predict_noise(cache_standard.x_t, cache_standard.t, c_cond)
        erased_uncond_pred = erased_predictor.predict_noise(cache_standard.x_t, cache_standard.t, uncond_cond)
        erased_guidance_delta = (erased_c_pred - erased_uncond_pred).detach()

    del ref_c_pred, ref_uncond_pred, erased_c_pred, erased_uncond_pred
    torch.cuda.empty_cache()

    # Guidance delta baseline MSE
    guidance_baseline_mse = float((erased_guidance_delta - ref_guidance_delta).pow(2).mean())
    print(f"Erased Baseline Guidance Delta MSE: {guidance_baseline_mse:.8f}")

    # Standard epsilon baseline MSE
    ref_scorer = ReferenceMatchScorer(reference_predictor=base_predictor, concept_cache=cache_standard, autocast_dtype=None)
    std_baseline_mse = ref_scorer.calibrate(erased_predictor)

    # Windowed epsilon baseline MSE
    window_ref_scorer = ReferenceMatchScorer(reference_predictor=base_predictor, concept_cache=cache_windowed, autocast_dtype=None)
    win_baseline_mse = window_ref_scorer.calibrate(erased_predictor)

    # Neutral cache for measuring neutral prediction drift
    neutral_latents = torch.randn(len(neutral_eval_prompts), 4, 64, 64, device=device, dtype=torch.float16)
    erased_neutral_cache = NoiseCache.build(base_predictor=erased_predictor, latents=neutral_latents, timesteps=None, cond=n_cond, seed=1337)
    torch.cuda.empty_cache()

    # Set up Optuna study on erased UNet
    provider = DictSubspaceProvider({
        layer: v_vangogh.to(dtype=torch.float16) for layer in cross_k_layers + cross_v_layers
    })
    targets = [
        ProjectionTarget(name=name, weight=erased_adapter.target_layers[name].module.weight, device=device)
        for name in cross_k_layers + cross_v_layers
    ]
    search_config = TrialConfig(projection_modes=("orthogonal",), target_projection="both", sampler="nsgaii", side="input")
    study = create_study(config=search_config, study_name="diagnostic-study")
    objective = build_objective(
        model=erased_unet,
        predictor=erased_predictor,
        subspace_provider=provider,
        concept_scorer=ref_scorer,
        neutral_cache=erased_neutral_cache,
        targets=targets,
        config=search_config,
    )
    print("Running 30 fresh search trials...")
    study.optimize(objective, n_trials=30)

    # Set up Generative Pipeline & Validators
    style_scorer = ClipStyleScorer(device=device)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, unet=erased_unet, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    validator = DeterministicGenerativeValidator(
        classifier=style_scorer,
        concept_prompts={"van gogh": heldout_prompts},
        neutral_prompts=neutral_eval_prompts[:4],
        config=ValidatorConfig(num_inference_steps=8, guidance_scale=3.0, batch_size=4, neutral_samples=4),
    )

    trials_data = []
    print(f"\nEvaluating 30 Trials on Generative Recovery & Alternative Proxies:")
    for idx, trial in enumerate(study.trials):
        alphas = trial.user_attrs["alphas"]
        edits = [
            LayerEdit(name=t.name, weight=t.weight, alpha=alphas[t.name], directions=provider.get(t.name), mode="orthogonal", side="input")
            for t in targets if alphas.get(t.name, 0.0) > 0.0
        ]

        with applied_edit(erased_unet, edits):
            # 1. Standard Proxy (Clipped)
            proxy_v1_clipped = float(trial.values[0])

            # 2. Standard Proxy (Unclipped)
            with torch.inference_mode():
                pred_std = erased_predictor.predict_noise(cache_standard.x_t, cache_standard.t, cache_standard.cond)
                mse_std = float((pred_std - cache_standard.base_pred).pow(2).mean())
                proxy_v1_unclipped = 1.0 - (mse_std / std_baseline_mse)

            # 3. Windowed Proxy (t in [100, 500], unclipped)
            with torch.inference_mode():
                pred_win = erased_predictor.predict_noise(cache_windowed.x_t, cache_windowed.t, cache_windowed.cond)
                mse_win = float((pred_win - cache_windowed.base_pred).pow(2).mean())
                proxy_windowed_unclipped = 1.0 - (mse_win / win_baseline_mse)

            # 4. Guidance Delta Proxy (unclipped)
            with torch.inference_mode():
                pred_c = pred_std
                pred_uncond = erased_predictor.predict_noise(cache_standard.x_t, cache_standard.t, uncond_cond)
                edited_guidance_delta = pred_c - pred_uncond
                mse_guidance = float((edited_guidance_delta - ref_guidance_delta).pow(2).mean())
                proxy_guidance_delta = 1.0 - (mse_guidance / guidance_baseline_mse)

            # 5. Generative Validation (Held-out Concept Prompts)
            pipe.unet = erased_unet
            trial_adapter = DiffusersAdapter(pipe, model_id=f"trial-{trial.number}", height=512, width=512)
            val_result = validator.validate(trial_adapter, "van gogh", num_samples=4, seed=42)
            real_concept_score = float(val_result.metrics["mean_confidence"])

            # 6. Specificity: Style Score on Neutral Prompts
            neutral_images = [trial_adapter.generate(p, seed=200 + i, num_inference_steps=8, guidance_scale=3.0) for i, p in enumerate(neutral_eval_prompts[:4])]
            neutral_scores = style_scorer.classify(neutral_images, "van gogh")
            mean_neutral_style = float(np.mean(neutral_scores))

        record = {
            "trial_id": trial.number,
            "proxy_v1_clipped": proxy_v1_clipped,
            "proxy_v1_unclipped": proxy_v1_unclipped,
            "proxy_windowed": proxy_windowed_unclipped,
            "proxy_guidance_delta": proxy_guidance_delta,
            "real_concept_recovery": real_concept_score,
            "real_neutral_style": mean_neutral_style,
            "block_range": trial.user_attrs.get("block_range"),
        }
        trials_data.append(record)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(study.trials):
            print(f"  [{idx + 1:2d}/30] Trial {trial.number:2d} | Real Concept: {real_concept_score:.4f} | Neutral Style: {mean_neutral_style:.4f} | Guidance Delta: {proxy_guidance_delta:.4f}")

    # Compute Spearman Rank Correlations against Real Generative Concept Recovery
    def safe_spearman(x, y):
        try:
            return spearman(x, y)
        except ValueError:
            return (float("nan"), 1.0)

    real_recs = [t["real_concept_recovery"] for t in trials_data]
    rho_v1_clip, p_v1_clip = safe_spearman([t["proxy_v1_clipped"] for t in trials_data], real_recs)
    rho_v1_unclip, p_v1_unclip = safe_spearman([t["proxy_v1_unclipped"] for t in trials_data], real_recs)
    rho_win, p_win = safe_spearman([t["proxy_windowed"] for t in trials_data], real_recs)
    rho_delta, p_delta = safe_spearman([t["proxy_guidance_delta"] for t in trials_data], real_recs)

    print("\n" + "=" * 70)
    print("PROXY VS REAL RECOVERY SPEARMAN RANK CORRELATION COMPARISON:")
    print("=" * 70)
    print(f"1. Proxy v1 (Raw epsilon, clipped [0, 1]):     rho = {rho_v1_clip:+.4f} (p = {p_v1_clip:.4e})")
    print(f"2. Proxy v1 (Raw epsilon, unclipped):          rho = {rho_v1_unclip:+.4f} (p = {p_v1_unclip:.4e})")
    print(f"3. Proxy v1 (Windowed t in [100, 500]):        rho = {rho_win:+.4f} (p = {p_win:.4e})")
    print(f"4. Proxy v2 (Guidance Delta epsilon(c)-eps(0)): rho = {rho_delta:+.4f} (p = {p_delta:.4e})")
    print("=" * 70)

    # Specificity Analysis
    all_neutral_styles = [t["real_neutral_style"] for t in trials_data]
    print(f"\nSpecificity Check (Van Gogh Style Classifier Score on Neutral Prompts):")
    print(f"  Mean across 30 edited models: {np.mean(all_neutral_styles):.4f} +/- {np.std(all_neutral_styles):.4f}")
    print(f"  Max across 30 edited models:  {np.max(all_neutral_styles):.4f}")
    print(f"  Min across 30 edited models:  {np.min(all_neutral_styles):.4f}")

    # -------------------------------------------------------------------------
    # CHECK 5: Test-Retest Reliability Ceiling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[DIAGNOSTIC 5] Test-Retest Reliability Ceiling on Real Recovery")
    print("=" * 70)
    retest_trials = study.trials[:6]
    run1_scores = []
    run2_scores = []

    for trial in retest_trials:
        alphas = trial.user_attrs["alphas"]
        edits = [
            LayerEdit(name=t.name, weight=t.weight, alpha=alphas[t.name], directions=provider.get(t.name), mode="orthogonal", side="input")
            for t in targets if alphas.get(t.name, 0.0) > 0.0
        ]
        with applied_edit(erased_unet, edits):
            pipe.unet = erased_unet
            adapter = DiffusersAdapter(pipe, model_id=f"retest-{trial.number}", height=512, width=512)
            r1 = validator.validate(adapter, "van gogh", num_samples=4, seed=42).metrics["mean_confidence"]
            r2 = validator.validate(adapter, "van gogh", num_samples=4, seed=999).metrics["mean_confidence"]
            run1_scores.append(float(r1))
            run2_scores.append(float(r2))

    retest_rho, retest_p = safe_spearman(run1_scores, run2_scores)
    print(f"Test-Retest Spearman rho (seed 42 vs seed 999, N=6): rho = {retest_rho:+.4f} (p = {retest_p:.4e})")

    return {
        "diff_summary": {cat: {"max": float(max(d)) if d else 0.0, "mean": float(np.mean(d)) if d else 0.0} for cat, d in category_diffs.items()},
        "top_non_cross_diffs": non_cross_diffs[:5],
        "svd_overlaps": {
            "van_gogh": vg_overlap,
            "gaussian_random_mean": float(np.mean(random_overlaps)),
            "unrelated_objects_mean": float(np.mean(object_overlaps)),
            "other_styles_mean": float(np.mean(style_overlaps)),
            "z_score_vs_text_null": z_score_text,
            "percentile_vs_text_null": percentile_text,
        },
        "spearman_comparisons": {
            "proxy_v1_clipped": {"rho": rho_v1_clip, "p": p_v1_clip},
            "proxy_v1_unclipped": {"rho": rho_v1_unclip, "p": p_v1_unclip},
            "proxy_windowed": {"rho": rho_win, "p": p_win},
            "proxy_guidance_delta": {"rho": rho_delta, "p": p_delta},
        },
        "specificity": {
            "neutral_style_mean": float(np.mean(all_neutral_styles)),
            "neutral_style_max": float(np.max(all_neutral_styles)),
        },
        "test_retest_ceiling": {
            "retest_rho": retest_rho,
            "retest_p": retest_p,
        },
    }


@app.local_entrypoint()
def main():
    print("Launching Heretic-DiT Empirical Diagnostics on Modal...")
    res = run_diagnostics.remote()
    print("\n=== DIAGNOSTIC RESULTS ===")
    print(json.dumps(res, indent=2))
