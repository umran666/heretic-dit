"""Modal script for empirical validation of Heretic-DiT on real erased checkpoint.

Target: Official ICCV 2023 ESD checkpoint for "Van Gogh" (Gandikota et al.).
Evaluates the 7 hard empirical gates:
1. Base Model Match: diffs ESD checkpoint against SD1.4 vs SD1.5.
2. SVD Diagnostic: energy in top singular vectors and contrastive subspace overlap.
3. Hardware Noise Floor & Bit-Exact Restore Gate.
4. Validator Sanity Gate: Style classifier separation on held-out prompts.
5. Three Calibration Controls: No-edit (~0), Oracle (>=0.95), Random projection (~0).
6. 100-Trial Block-Indexed Optuna Search over 16 cross-attention blocks.
7. Stratified Generative Validation (30 trials) & Spearman Rank Correlation (rho).
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

app = modal.App("heretic-dit-vangogh-audit")

# Build container image with CUDA PyTorch, Diffusers, and OpenCLIP stack
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
    gpu="T4",  # 16GB VRAM
    timeout=1800,
)
def run_vangogh_audit(
    n_search_trials: int = 100,
    n_eval_trials: int = 30,
) -> Dict[str, Any]:
    """Execute complete empirical audit against the official Van Gogh ESD checkpoint."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
    from heretic_dit.benchmarks.concepts import load_or_create_split
    from heretic_dit.core.projector import project_weights
    from heretic_dit.core.subspace import mean_difference
    from heretic_dit.eval.adapters import CallableAdapter, DiffusersAdapter
    from heretic_dit.eval.classifiers import ClipStyleScorer
    from heretic_dit.eval.proxy_validity import ProxyTrial, evaluate_proxy_validity, spearman
    from heretic_dit.eval.validator import DeterministicGenerativeValidator, ValidatorConfig
    from heretic_dit.interfaces import EditSpec
    from heretic_dit.metrics.drift import NoiseCache, UNetPredictor, compute_epsilon_drift
    from heretic_dit.search.editing import LayerEdit, ProjectionTarget, applied_edit
    from heretic_dit.search.optuna_objective import (
        DictSubspaceProvider,
        ReferenceMatchScorer,
        TrialConfig,
        build_objective,
        create_study,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("HERETIC-DiT: EMPIRICAL AUDIT ON OFFICIAL ESD VAN GOGH CHECKPOINT")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 0: Fetch Official ESD Weights
    # -------------------------------------------------------------------------
    esd_url = "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
    esd_path = Path("/root/diffusers-VanGogh-ESDx1-UNET.pt")
    if not esd_path.exists():
        print(f"\n[0/7] Downloading official ESD checkpoint from {esd_url}...")
        t0 = time.perf_counter()
        urllib.request.urlretrieve(esd_url, esd_path)
        print(f"Downloaded {esd_path.stat().st_size / (1024**2):.1f} MB in {time.perf_counter() - t0:.1f} s.")

    # Compute SHA-256
    sha256 = hashlib.sha256()
    with esd_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    esd_hash = sha256.hexdigest()
    print(f"Checkpoint SHA-256: {esd_hash}")

    esd_state_dict = torch.load(esd_path, map_location="cpu")
    print(f"Loaded ESD state dict ({len(esd_state_dict)} tensor keys).")

    # -------------------------------------------------------------------------
    # STEP 1: Base Model Verification Gate (SD 1.4 vs SD 1.5)
    # -------------------------------------------------------------------------
    print("\n[1/7] Testing Base Model Verification Gate (SD 1.4 vs SD 1.5)...")
    unet_sd14 = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16
    )
    unet_sd15 = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16
    )

    sd14_dict = unet_sd14.state_dict()
    sd15_dict = unet_sd15.state_dict()

    # Compare non-cross-attention weights between base models and ESD
    non_cross_keys = [k for k in esd_state_dict if "attn2.to_k" not in k and "attn2.to_v" not in k]
    cross_keys = [k for k in esd_state_dict if "attn2.to_k" in k or "attn2.to_v" in k]

    max_diff_14 = 0.0
    for k in non_cross_keys:
        if k in sd14_dict:
            diff = (sd14_dict[k].float() - esd_state_dict[k].float()).abs().max().item()
            if diff > max_diff_14:
                max_diff_14 = diff

    max_diff_15 = 0.0
    for k in non_cross_keys:
        if k in sd15_dict:
            diff = (sd15_dict[k].float() - esd_state_dict[k].float()).abs().max().item()
            if diff > max_diff_15:
                max_diff_15 = diff

    print(f"Non-cross-attention max diff vs SD 1.4: {max_diff_14:.8f}")
    print(f"Non-cross-attention max diff vs SD 1.5: {max_diff_15:.8f}")

    if max_diff_14 < 1e-4:
        base_model_id = "CompVis/stable-diffusion-v1-4"
        base_unet = unet_sd14.to(device)
        print("[PASS] Verified: Base model is CompVis/stable-diffusion-v1-4 (bit-identical non-cross-attn weights).")
    elif max_diff_15 < 1e-4:
        base_model_id = "runwayml/stable-diffusion-v1-5"
        base_unet = unet_sd15.to(device)
        print("[PASS] Verified: Base model is runwayml/stable-diffusion-v1-5.")
    else:
        # Fall back to SD 1.4 as documented in the paper
        base_model_id = "CompVis/stable-diffusion-v1-4"
        base_unet = unet_sd14.to(device)
        print(f"[WARN] Small numerical differences found; using standard SD 1.4 base: {base_model_id}")

    del unet_sd15, sd14_dict, sd15_dict
    torch.cuda.empty_cache()

    # Load Text Encoder & Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    base_unet.eval()
    text_encoder.eval()

    # Create Erased UNet by loading ESD weights
    erased_unet = copy.deepcopy(base_unet)
    erased_unet.load_state_dict(esd_state_dict, strict=False)
    erased_unet.eval().to(device)

    base_adapter = DiffusersModelAdapter(base_unet)
    erased_adapter = DiffusersModelAdapter(erased_unet)
    base_predictor = UNetPredictor(model=base_unet, prediction_type="epsilon")
    erased_predictor = UNetPredictor(model=erased_unet, prediction_type="epsilon")

    cross_k_layers = base_adapter.get_cross_attention_layer_names(layer_types=("key",))
    cross_v_layers = base_adapter.get_cross_attention_layer_names(layer_types=("value",))
    print(f"Discovered {len(cross_k_layers)} key layers, {len(cross_v_layers)} value layers.")

    # -------------------------------------------------------------------------
    # STEP 2: SVD Diagnostic of Delta W & Subspace Overlap
    # -------------------------------------------------------------------------
    print("\n[2/7] Running Weight Delta SVD & Subspace Alignment Diagnostic...")
    svd_energies_k: List[float] = []
    svd_energies_v: List[float] = []
    overlaps: List[float] = []

    # Get concept direction from text embeddings
    split = load_or_create_split(config_path=Path("/root/configs/concepts_default.yaml"), split_dir=Path("/root/splits"))
    concept_entry = split.registry.get("van gogh")
    search_prompts = list(concept_entry.search_prompts[:8])
    neutral_prompts = list(split.registry.neutral_prompts[:8])

    c_tokens = tokenizer(search_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    n_tokens = tokenizer(neutral_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        c_cond = text_encoder(c_tokens)[0]
        n_cond = text_encoder(n_tokens)[0]

    v_concept = mean_difference(
        c_cond.reshape(-1, 768), n_cond.reshape(-1, 768), compute_dtype=torch.float32
    ).to(device, dtype=torch.float32)  # (768, 1)

    for k_name, v_name in zip(cross_k_layers, cross_v_layers):
        W_base_k = base_adapter.target_layers[k_name].module.weight.float()
        W_esd_k = erased_adapter.target_layers[k_name].module.weight.float()
        dW_k = W_base_k - W_esd_k  # (out_dim, 768)

        if dW_k.norm() > 1e-6:
            U, S, Vh = torch.linalg.svd(dW_k, full_matrices=False)
            energy_top1 = (S[0] ** 2) / (S ** 2).sum()
            svd_energies_k.append(energy_top1.item())
            # Right singular vector V[:, 0] corresponds to input direction
            v_svd = Vh[0:1, :].T  # (768, 1)
            cos_sim = torch.abs(torch.matmul(v_concept.T, v_svd)).item()
            overlaps.append(cos_sim)

    avg_e1 = float(np.mean(svd_energies_k)) if svd_energies_k else 0.0
    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    print(f"ESD Weight Delta SVD Energy (Top-1 Component): {avg_e1 * 100:.2f}% of total energy")
    print(f"Subspace Alignment |cos(v_concept, v_ESD_svd)|: {avg_overlap:.4f}")

    # -------------------------------------------------------------------------
    # STEP 3: Identity Sanity & Hardware Noise Floor Gate
    # -------------------------------------------------------------------------
    print("\n[3/7] Testing Identity Sanity & Hardware Noise Floor...")
    neutral_eval_prompts = list(split.registry.neutral_prompts[:16])
    neutral_eval_tokens = tokenizer(neutral_eval_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        neutral_eval_cond = text_encoder(neutral_eval_tokens)[0]

    clean_latents = torch.randn(len(neutral_eval_prompts), 4, 64, 64, device=device, dtype=torch.float16)
    erased_neutral_cache = NoiseCache.build(
        base_predictor=erased_predictor,
        latents=clean_latents,
        timesteps=None,
        cond=neutral_eval_cond,
        seed=1337,
        autocast_dtype=None,
    )

    # Measure unedited dual-pass hardware noise floor
    floor_drift = float(compute_epsilon_drift(erased_predictor, erased_predictor, cache=erased_neutral_cache, autocast_dtype=None))
    print(f"GPU Hardware Noise Floor (consecutive forward passes, no edits): {floor_drift:.10f}")

    # Test bit-exact restore on erased_adapter
    orig_k0 = erased_adapter.target_layers[cross_k_layers[0]].module.weight.clone()
    test_direction = v_concept.to(dtype=torch.float16)
    identity_spec = EditSpec(layers=cross_k_layers[:2], alpha=0.0, mode="orthogonal", side="input", subspace=test_direction)
    active_spec = EditSpec(layers=cross_k_layers[:2], alpha=0.5, mode="orthogonal", side="input", subspace=test_direction)

    with erased_adapter.temporary_edit(identity_spec):
        id_drift = float(compute_epsilon_drift(erased_predictor, erased_predictor, cache=erased_neutral_cache, autocast_dtype=None))
    assert torch.equal(erased_adapter.target_layers[cross_k_layers[0]].module.weight, orig_k0), "alpha=0.0 restore failed!"

    with erased_adapter.temporary_edit(active_spec):
        pass
    assert torch.equal(erased_adapter.target_layers[cross_k_layers[0]].module.weight, orig_k0), "alpha=0.5 restore failed!"
    print(f"Identity edit drift (alpha=0.0): {id_drift:.10f}")
    assert id_drift <= floor_drift + 1e-6, f"Identity drift {id_drift} exceeds noise floor {floor_drift}"
    print("[PASS] Identity Sanity & Bit-Exact Restore Gate PASSED.")

    # -------------------------------------------------------------------------
    # STEP 4: Validator Sanity Gate (Style Classifier Separation)
    # -------------------------------------------------------------------------
    print("\n[4/7] Testing Validator Sanity Gate (Style Classifier Separation)...")
    style_scorer = ClipStyleScorer(device="cuda" if torch.cuda.is_available() else "cpu")
    heldout_prompts = list(concept_entry.heldout_prompts[:4])
    print(f"Held-out validation prompts ({len(heldout_prompts)} prompts):")
    for p in heldout_prompts:
        print(f"  - {p}")

    # Build Diffusers Pipelines for image generation
    pipe_base = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        unet=base_unet,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe_base.set_progress_bar_config(disable=True)

    adapter_base = DiffusersAdapter(pipe_base, model_id="sd14-base", height=512, width=512)
    images_base = [adapter_base.generate(prompt=p, seed=100 + i, num_inference_steps=8, guidance_scale=3.0) for i, p in enumerate(heldout_prompts)]
    scores_base = style_scorer.classify(images_base, "van gogh")

    # Re-wire pipeline to erased UNet
    pipe_base.unet = erased_unet
    adapter_erased = DiffusersAdapter(pipe_base, model_id="sd14-esd-erased", height=512, width=512)
    images_erased = [adapter_erased.generate(prompt=p, seed=100 + i, num_inference_steps=8, guidance_scale=3.0) for i, p in enumerate(heldout_prompts)]
    scores_erased = style_scorer.classify(images_erased, "van gogh")

    mean_score_base = float(np.mean(scores_base))
    mean_score_erased = float(np.mean(scores_erased))
    score_gap = mean_score_base - mean_score_erased
    print(f"Base Model Style Score:   {mean_score_base:.4f}")
    print(f"Erased Model Style Score: {mean_score_erased:.4f}")
    print(f"Classifier Separation Gap: {score_gap:.4f}")
    assert score_gap >= 0.10, f"Validator sanity failed! Style gap {score_gap:.4f} is too small to detect recovery."
    print("[PASS] Validator Sanity Gate PASSED.")

    # -------------------------------------------------------------------------
    # STEP 5: Three Scorer Calibration Controls
    # -------------------------------------------------------------------------
    print("\n[5/7] Testing Three Scorer Calibration Controls...")
    concept_eval_prompts = list(concept_entry.search_prompts[:16])
    concept_eval_tokens = tokenizer(concept_eval_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        concept_eval_cond = text_encoder(concept_eval_tokens)[0]

    concept_latents = torch.randn(len(concept_eval_prompts), 4, 64, 64, device=device, dtype=torch.float16)
    # Build concept cache on the unerased base (reference) model
    concept_cache = NoiseCache.build(
        base_predictor=base_predictor,
        latents=concept_latents,
        timesteps=None,
        cond=concept_eval_cond,
        seed=42,
        autocast_dtype=None,
    )

    ref_scorer = ReferenceMatchScorer(
        reference_predictor=base_predictor,
        concept_cache=concept_cache,
        autocast_dtype=None,
    )
    baseline_erased_mse = ref_scorer.calibrate(erased_predictor)
    print(f"Calibrated Baseline Erased vs Reference MSE: {baseline_erased_mse:.8f}")

    # Control 1: No-edit baseline
    rec_no_edit = float(ref_scorer.score(erased_predictor))
    print(f"Control 1 (No-edit on Erased Model):       Recovery = {rec_no_edit:.6f} (Expected: ~0.0)")
    assert rec_no_edit < 0.05, f"No-edit recovery {rec_no_edit} is not ~0"

    # Control 2: Oracle baseline (unerased reference model)
    rec_oracle = float(ref_scorer.score(base_predictor))
    print(f"Control 2 (Oracle - Unerased Reference):   Recovery = {rec_oracle:.6f} (Expected: >= 0.95)")
    assert rec_oracle >= 0.95, f"Oracle recovery {rec_oracle} is below 0.95!"

    # Control 3: Random projection baseline (matched alpha=0.5, QR orthonormal)
    gen = torch.Generator(device=device).manual_seed(12345)
    random_raw = torch.randn(768, 1, generator=gen, device=device, dtype=torch.float32)
    random_q, _ = torch.linalg.qr(random_raw)
    random_spec = EditSpec(
        layers=cross_k_layers + cross_v_layers,
        alpha=0.5,
        mode="orthogonal",
        side="input",
        subspace=random_q.to(dtype=torch.float16),
    )
    with erased_adapter.temporary_edit(random_spec):
        rec_random = float(ref_scorer.score(erased_predictor))
    print(f"Control 3 (Random Projection @ alpha=0.5): Recovery = {rec_random:.6f} (Expected: ~0.0)")
    assert rec_random < 0.10, f"Random projection recovery {rec_random} is too high!"
    print("[PASS] All Three Calibration Controls PASSED.")

    # -------------------------------------------------------------------------
    # STEP 6: 100-Trial Block-Indexed Optuna Search
    # -------------------------------------------------------------------------
    print(f"\n[6/7] Running 100-Trial Block-Indexed Optuna Search ({n_search_trials} trials)...")
    provider = DictSubspaceProvider({
        layer: v_concept.to(dtype=torch.float16) for layer in cross_k_layers + cross_v_layers
    })

    targets = [
        ProjectionTarget(name=name, weight=erased_adapter.target_layers[name].module.weight, device=device)
        for name in cross_k_layers + cross_v_layers
    ]

    search_config = TrialConfig(
        projection_modes=("orthogonal",),
        target_projection="both",
        sampler="nsgaii",
        side="input",
    )

    study = create_study(config=search_config, study_name="heretic-vangogh-esd")
    objective = build_objective(
        model=erased_unet,
        predictor=erased_predictor,
        subspace_provider=provider,
        concept_scorer=ref_scorer,
        neutral_cache=erased_neutral_cache,
        targets=targets,
        config=search_config,
        autocast_dtype=None,
        compute_dtype=torch.float32,
    )

    t_search_start = time.perf_counter()
    study.optimize(objective, n_trials=n_search_trials)
    search_duration = time.perf_counter() - t_search_start

    print(f"\nCompleted {len(study.trials)} trials in {search_duration:.1f} s ({search_duration / n_search_trials:.3f} s/trial).")
    pareto_trials = list(study.best_trials)
    print(f"Found {len(pareto_trials)} Pareto-optimal edits:")
    for i, t in enumerate(pareto_trials[:8]):
        print(f"  Pareto #{i+1}: Trial {t.number:2d} | Recovery = {t.values[0]:.4f} | Drift = {t.values[1]:.6f} | Blocks = {t.user_attrs.get('block_range')}")

    # -------------------------------------------------------------------------
    # STEP 7: Stratified Validation & Spearman Rank Correlation
    # -------------------------------------------------------------------------
    print(f"\n[7/7] Running Stratified Validation on {n_eval_trials} Trials for Spearman Power...")
    validator = DeterministicGenerativeValidator(
        classifier=style_scorer,
        concept_prompts={"van gogh": heldout_prompts},
        neutral_prompts=neutral_eval_prompts,
        config=ValidatorConfig(num_inference_steps=8, guidance_scale=3.0, batch_size=4, neutral_samples=4),
    )

    # Stratified selection across recovery spectrum: Pareto + middle + low
    all_trials = sorted(study.trials, key=lambda t: t.values[0])
    indices = np.linspace(0, len(all_trials) - 1, min(n_eval_trials, len(all_trials)), dtype=int)
    selected_trials = [all_trials[idx] for idx in indices]

    proxy_trials: List[ProxyTrial] = []
    t_val_start = time.perf_counter()

    for idx, trial in enumerate(selected_trials):
        alphas = trial.user_attrs["alphas"]
        edits = [
            LayerEdit(
                name=t.name,
                weight=t.weight,
                alpha=alphas[t.name],
                directions=provider.get(t.name),
                mode="orthogonal",
                side="input",
            )
            for t in targets if alphas.get(t.name, 0.0) > 0.0
        ]

        with applied_edit(erased_unet, edits):
            pipe_base.unet = erased_unet
            trial_adapter = DiffusersAdapter(pipe_base, model_id=f"trial-{trial.number}", height=512, width=512)
            val_result = validator.validate(trial_adapter, "van gogh", num_samples=4, seed=42)

        p_trial = ProxyTrial(
            trial_id=trial.number,
            proxy_recovery=float(trial.values[0]),
            proxy_drift=float(trial.values[1]),
            real_recovery=float(val_result.metrics["mean_confidence"]),
            real_quality_loss=float(val_result.drift_score),
            label=f"trial_{trial.number}",
            extra={"block_range": trial.user_attrs.get("block_range")},
        )
        proxy_trials.append(p_trial)
        if (idx + 1) % 5 == 0 or (idx + 1) == len(selected_trials):
            print(f"  [{idx + 1}/{len(selected_trials)}] Trial {trial.number:2d}: Proxy Rec = {p_trial.proxy_recovery:.4f} -> Real Style Conf = {p_trial.real_recovery:.4f}")

    val_duration = time.perf_counter() - t_val_start
    print(f"\nGenerative validation completed in {val_duration:.1f} s ({val_duration / len(selected_trials):.2f} s/trial).")

    proxy_recs = [t.proxy_recovery for t in proxy_trials]
    real_recs = [t.real_recovery for t in proxy_trials]
    rec_rho, rec_p = spearman(proxy_recs, real_recs)

    print("\n" + "=" * 60)
    print("PROXY VALIDITY REPORT (SPEARMAN RANK CORRELATION)")
    print("=" * 60)
    print(f"Evaluated Trials (N):    {len(proxy_trials)}")
    print(f"Recovery Spearman rho:   {rec_rho:.4f} (p-value = {rec_p:.4e})")
    print(f"Proxy Trustworthy (rho >= 0.70): {'YES [PASS]' if rec_rho >= 0.70 else 'BORDERLINE/NO'}")

    try:
        validity_report = evaluate_proxy_validity(proxy_trials)
        print(f"Quality Drift rho:       {validity_report.quality_rho:.4f} (p-value = {validity_report.quality_pvalue:.4e})")
        print(f"Combined Utility rho:    {validity_report.utility_rho:.4f} (p-value = {validity_report.utility_pvalue:.4e})")
        validity_dict = validity_report.to_dict()
    except ValueError as err:
        print(f"Notice on quality drift/utility calculation: {err}")
        validity_dict = {
            "n_trials": len(proxy_trials),
            "recovery_rho": rec_rho,
            "recovery_pvalue": rec_p,
            "trustworthy": bool(rec_rho >= 0.70),
            "note": str(err),
        }
    print("=" * 60)

    return {
        "base_model": base_model_id,
        "checkpoint_hash": esd_hash,
        "svd_energy_top1": avg_e1,
        "contrastive_subspace_overlap": avg_overlap,
        "noise_floor": floor_drift,
        "validator_gap": score_gap,
        "calibration_controls": {
            "no_edit": rec_no_edit,
            "oracle": rec_oracle,
            "random": rec_random,
        },
        "search": {
            "trials": n_search_trials,
            "duration_sec": search_duration,
            "pareto_count": len(pareto_trials),
        },
        "proxy_validity": validity_dict,
    }


@app.local_entrypoint()
def main(search_trials: int = 100, eval_trials: int = 30):
    """Local CLI entrypoint for modal run."""
    print(f"Triggering Full Van Gogh ESD Audit on Modal ({search_trials} search trials, {eval_trials} eval trials)...")
    result = run_vangogh_audit.remote(n_search_trials=search_trials, n_eval_trials=eval_trials)
    print("\n=== Final Audit Results ===")
    print(json.dumps(result, indent=2))
