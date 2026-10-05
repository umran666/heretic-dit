"""Modal script for empirical validation of Heretic-DiT on real GPU hardware.

Tests:
1. Real SD 1.5 model loading (runwayml/stable-diffusion-v1-5 in fp16).
2. Identity Sanity Gate: alpha = 0.0 yields drift == 0.0 on real noisy latents.
3. Optuna Trial Latency: timing single-trial drift evaluations on an A10G GPU.
4. Concept Subspace Extraction & Multi-Objective Search for "parachute".
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("heretic-dit-audit")

# Build container image with CUDA PyTorch and Diffusers stack
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
    )
    .add_local_python_source("heretic_dit")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("splits", remote_path="/root/splits")
)


@app.function(
    image=image,
    gpu="T4",  # 16GB VRAM (supported on starter tier)
    timeout=1200,
)
def run_real_model_audit(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    concept_name: str = "parachute",
    n_trials: int = 30,
) -> Dict[str, Any]:
    """Execute empirical audit on real SD1.5 weights on an A10G GPU."""
    import torch
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    from heretic_dit.architectures.diffusers_adapter import DiffusersModelAdapter
    from heretic_dit.benchmarks.concepts import load_or_create_split
    from heretic_dit.core.subspace import mean_difference
    from heretic_dit.interfaces import EditSpec
    from heretic_dit.metrics.drift import NoiseCache, UNetPredictor, compute_epsilon_drift
    from heretic_dit.search.optuna_objective import (
        DictSubspaceProvider,
        DenoisingLossScorer,
        TrialConfig,
        build_objective,
        create_study,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Heretic-DiT GPU Audit ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    # 1. Load Real Models in FP16
    print(f"\n[1/4] Loading models from {model_id}...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)
    unet.eval()
    text_encoder.eval()

    adapter = DiffusersModelAdapter(unet)
    predictor = UNetPredictor(model=unet, prediction_type="epsilon")

    cross_k_layers = adapter.get_cross_attention_layer_names(layer_types=("key",))
    cross_v_layers = adapter.get_cross_attention_layer_names(layer_types=("value",))
    print(f"Discovered {len(cross_k_layers)} key layers, {len(cross_v_layers)} value layers.")

    # 2. Identity Sanity Gate (alpha=0 -> drift=0)
    print("\n[2/4] Testing Identity Sanity Gate (alpha=0.0)...")
    split = load_or_create_split(config_path=Path("/root/configs/concepts_default.yaml"), split_dir=Path("/root/splits"))
    neutral_prompts = list(split.registry.neutral_prompts[:16])

    # Encode neutral prompts
    neutral_tokens = tokenizer(neutral_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        neutral_cond = text_encoder(neutral_tokens)[0]

    # Build real NoiseCache
    clean_latents = torch.randn(len(neutral_prompts), 4, 64, 64, device=device, dtype=torch.float16)
    cache = NoiseCache.build(
        base_predictor=predictor,
        latents=clean_latents,
        timesteps=None,  # Stratified low/mid/high bins
        cond=neutral_cond,
        seed=1337,
        autocast_dtype=None,
    )

    # Apply identity edit (alpha = 0.0)
    dummy_direction = torch.randn(768, 1, device=device, dtype=torch.float16)
    dummy_direction = dummy_direction / dummy_direction.norm()
    identity_spec = EditSpec(
        layers=cross_k_layers[:2],
        alpha=0.0,
        mode="orthogonal",
        side="input",
        subspace=dummy_direction,
    )

    with adapter.temporary_edit(identity_spec):
        identity_drift = float(compute_epsilon_drift(predictor, predictor, cache=cache))

    print(f"Identity drift (alpha=0.0): {identity_drift:.8f}")
    assert identity_drift < 1e-6, f"Identity sanity failed! Drift was {identity_drift}"
    print("[PASS] Identity Sanity Gate PASSED.")

    # 3. Trial Latency Benchmark Gate
    print("\n[3/4] Benchmarking Optuna Single-Trial Latency...")
    test_spec = EditSpec(
        layers=cross_k_layers[:8],
        alpha=0.5,
        mode="orthogonal",
        side="input",
        subspace=dummy_direction,
    )

    latencies: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        with adapter.temporary_edit(test_spec):
            _ = compute_epsilon_drift(predictor, predictor, cache=cache)
        latencies.append(time.perf_counter() - t0)

    avg_latency = sum(latencies) / len(latencies)
    print(f"Trial Latency (16 samples, 8 layers): {avg_latency:.3f} s (Target: 1-2 s)")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**2):.1f} MB")
    assert avg_latency < 3.0, f"Trial latency too slow: {avg_latency:.2f} s"
    print("[PASS] Latency Benchmark Gate PASSED.")

    # 4. Contrastive Concept Subspace & Optuna Search for "parachute"
    print(f"\n[4/4] Running Subspace Extraction & Optuna Search for '{concept_name}'...")
    concept_entry = split.registry[concept_name]
    concept_prompts = list(split.search_prompts(concept_name)[:8])

    concept_tokens = tokenizer(concept_prompts, padding=True, truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        concept_cond = text_encoder(concept_tokens)[0]

    # Extract concept direction via mean difference
    concept_flat = concept_cond.reshape(-1, 768)
    neutral_flat = neutral_cond.reshape(-1, 768)
    v_concept = mean_difference(concept_flat, neutral_flat, compute_dtype=torch.float32).to(device, dtype=torch.float16)

    subspace_provider = DictSubspaceProvider({
        layer: v_concept for layer in cross_k_layers + cross_v_layers
    })

    # Concept noise cache for recovery scoring
    concept_latents = torch.randn(len(concept_prompts), 4, 64, 64, device=device, dtype=torch.float16)
    concept_cache = NoiseCache.build(
        base_predictor=predictor,
        latents=concept_latents,
        timesteps=None,
        cond=concept_cond,
        seed=42,
        autocast_dtype=None,
    )
    scorer = DenoisingLossScorer(concept_cache=concept_cache)

    config = TrialConfig(
        concept=concept_name,
        layers=tuple(cross_k_layers[:16]),
        max_alpha=1.0,
        n_trials=n_trials,
        seed=42,
    )

    study = create_study(study_name=f"audit-{concept_name}", seed=config.seed)
    objective = build_objective(
        base_predictor=predictor,
        neutral_cache=cache,
        subspace_provider=subspace_provider,
        concept_scorer=scorer,
        config=config,
    )

    t_search_start = time.perf_counter()
    study.optimize(objective, n_trials=config.n_trials)
    search_duration = time.perf_counter() - t_search_start

    pareto_trials = [t for t in study.best_trials]
    print(f"\nStudy completed {len(study.trials)} trials in {search_duration:.2f} s ({search_duration/n_trials:.3f} s/trial).")
    print(f"Found {len(pareto_trials)} Pareto-optimal edits:")
    for i, t in enumerate(pareto_trials[:5]):
        # Objective 0: Maximize recovery, Objective 1: Minimize drift
        rec = t.values[0]
        drift = t.values[1]
        print(f"  Pareto #{i+1}: Recovery = {rec:.4f}, Neutral Drift = {drift:.6f}")

    return {
        "device": str(device),
        "model_id": model_id,
        "concept": concept_name,
        "n_trials": n_trials,
        "search_duration_sec": search_duration,
        "avg_trial_sec": search_duration / n_trials,
        "identity_drift": identity_drift,
        "pareto_trials_count": len(pareto_trials),
        "best_pareto": [
            {"trial_id": t.number, "recovery": t.values[0], "drift": t.values[1], "params": t.params}
            for t in pareto_trials[:5]
        ],
    }


@app.local_entrypoint()
def main(concept: str = "parachute", trials: int = 30):
    """Local CLI entrypoint for modal run."""
    print(f"Triggering Modal GPU audit for concept '{concept}' with {trials} trials...")
    result = run_real_model_audit.remote(concept_name=concept, n_trials=trials)
    print("\n=== Remote Audit Result ===")
    import json
    print(json.dumps(result, indent=2))
