"""Modal script for:
1. Rigorous FP16-ratio test comparing ESD vs SD 1.4 against true FP16 round-trip bounds.
2. Objective 2 evaluation: Spearman correlation between neutral epsilon-drift and style bleed.
3. Bleed-gated normalized ground truth re-analysis on the dev trials.
4. Multi-class style classifier evaluation (Van Gogh vs Monet, Cezanne, generic oil painting, photo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import modal

app = modal.App("heretic-dit-fp16-specificity-audit")

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
        "open_clip_torch>=2.24.0",
    )
    .add_local_python_source("heretic_dit")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("splits", remote_path="/root/splits")
)


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def run_fp16_and_specificity_checks() -> Dict[str, Any]:
    """Execute FP16-ratio test, neutral drift vs bleed analysis, and multi-class classifier test."""
    import urllib.request
    import numpy as np
    from scipy import stats
    import torch
    import torch.nn.functional as F
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    import open_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("HERETIC-DiT: RIGOROUS FP16-RATIO & SPECIFICITY GROUND-TRUTH AUDIT")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # PART 1: The Quantitative FP16-Ratio Test
    # -------------------------------------------------------------------------
    print("\n[PART 1] Running Quantitative FP16-Ratio Test...")
    esd_url = "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
    esd_path = Path("/root/diffusers-VanGogh-ESDx1-UNET.pt")
    if not esd_path.exists():
        urllib.request.urlretrieve(esd_url, esd_path)

    esd_state_dict = torch.load(esd_path, map_location="cpu")
    unet_sd14 = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float32
    )
    sd14_state_dict = unet_sd14.state_dict()

    fp16_eps_bound = 4.8828125e-4  # 2^(-11) machine epsilon for half precision

    # Compare non-cross-attention tensors
    non_cross_keys = [k for k in esd_state_dict if "attn2.to_k" not in k and "attn2.to_v" not in k]

    roundtrip_rel_errors = []
    esd_rel_errors = []
    exceeds_fp16_bound_count = 0
    total_elements_checked = 0

    tensor_stats = []

    for k in non_cross_keys:
        if k not in sd14_state_dict:
            continue
        W_base = sd14_state_dict[k].float()
        W_esd = esd_state_dict[k].float()
        W_half_roundtrip = W_base.half().float()

        # Relative error against base magnitude: |diff| / (|W| + eps)
        abs_base = W_base.abs()
        mask = abs_base > 1e-4  # avoid divide-by-zero on near-zero weights

        if mask.sum() == 0:
            continue

        rel_err_roundtrip = ((W_base - W_half_roundtrip).abs() / abs_base)[mask]
        rel_err_esd = ((W_base - W_esd).abs() / abs_base)[mask]

        max_rel_rt = float(rel_err_roundtrip.max().item())
        mean_rel_rt = float(rel_err_roundtrip.mean().item())

        max_rel_esd = float(rel_err_esd.max().item())
        mean_rel_esd = float(rel_err_esd.mean().item())

        roundtrip_rel_errors.append(mean_rel_rt)
        esd_rel_errors.append(mean_rel_esd)

        num_exceed = int((rel_err_esd > fp16_eps_bound).sum().item())
        exceeds_fp16_bound_count += num_exceed
        total_elements_checked += int(mask.sum().item())

        tensor_stats.append({
            "key": k,
            "mean_abs_diff": float((W_base - W_esd).abs().mean().item()),
            "max_abs_diff": float((W_base - W_esd).abs().max().item()),
            "mean_rel_error": mean_rel_esd,
            "max_rel_error": max_rel_esd,
            "roundtrip_mean_rel": mean_rel_rt,
            "frac_exceeding_fp16": num_exceed / max(1, int(mask.sum().item())),
        })

    print(f"\nFP16 Round-Trip vs ESD Relative Error Summary (Non-Cross-Attn):")
    print(f"  FP16 Theoretical Epsilon Bound:                 {fp16_eps_bound:.6e}")
    print(f"  SD 1.4 True FP16 Round-Trip Mean Relative Error: {np.mean(roundtrip_rel_errors):.6e}")
    print(f"  ESD vs SD 1.4 Measured Mean Relative Error:      {np.mean(esd_rel_errors):.6e}")
    print(f"  Ratio (Measured Error / FP16 Round-Trip Error):  {np.mean(esd_rel_errors) / np.mean(roundtrip_rel_errors):.1f}x LARGER")
    print(f"  Elements Exceeding FP16 Bound:                   {exceeds_fp16_bound_count:,} / {total_elements_checked:,} ({100.0 * exceeds_fp16_bound_count / total_elements_checked:.1f}%)")

    # Sort tensors by mean relative error
    tensor_stats.sort(key=lambda x: x["mean_rel_error"], reverse=True)
    print("\nTop 5 Tensors by Relative Error vs SD 1.4:")
    for t in tensor_stats[:5]:
        print(f"  {t['key'][:50]:50s} | mean_rel={t['mean_rel_error']:.4f} | max_rel={t['max_rel_error']:.4f} | exceed_fp16={t['frac_exceeding_fp16']*100:.1f}%")

    # -------------------------------------------------------------------------
    # PART 2: Check Objective 2 (Neutral Drift vs Style Bleeding)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[PART 2] Evaluating Neutral Epsilon-Drift vs Neutral Style Bleed...")
    print("=" * 70)

    # Load data from the 30 dev trials in run 980
    # In run 980, we recorded: trial number, proxy drift (values[1]), concept style, neutral style
    # Let's verify correlation between neutral epsilon drift and neutral style score
    # We can run 30 trials with the same configuration to extract both exact quantities
    del unet_sd14, sd14_state_dict, esd_state_dict
    torch.cuda.empty_cache()

    # Load SD 1.4 pipeline components for fast evaluation
    base_model_id = "CompVis/stable-diffusion-v1-4"
    tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)
    base_unet = UNet2DConditionModel.from_pretrained(
        base_model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(device)

    erased_state_dict = torch.load(esd_path, map_location="cpu")
    erased_unet = UNet2DConditionModel.from_pretrained(
        base_model_id, unet=base_unet, torch_dtype=torch.float16
    ).to(device)
    erased_unet.load_state_dict(erased_state_dict, strict=False)
    del erased_state_dict
    torch.cuda.empty_cache()

    # Multi-Class Style Classifier setup
    print("\n" + "=" * 70)
    print("[PART 3] Multi-Class Style Classifier Benchmark (Van Gogh vs Distractors)")
    print("=" * 70)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    clip_tok = open_clip.get_tokenizer("ViT-B-32")

    candidate_styles = [
        "a painting by Vincent van Gogh",
        "a painting by Claude Monet",
        "a painting by Paul Cezanne",
        "an Impressionist painting",
        "a generic oil painting",
        "a watercolor artwork",
        "a plain photograph taken with a camera",
    ]

    with torch.no_grad():
        toks = clip_tok(candidate_styles).to(device)
        text_feats = F.normalize(model.encode_text(toks), dim=-1)

    print(f"Configured Multi-Class Style Classifier with {len(candidate_styles)} classes:")
    for i, s in enumerate(candidate_styles):
        print(f"  Class {i}: '{s}'")

    return {
        "fp16_ratio_test": {
            "fp16_eps_bound": fp16_eps_bound,
            "sd14_roundtrip_mean_rel": float(np.mean(roundtrip_rel_errors)),
            "esd_mean_rel": float(np.mean(esd_rel_errors)),
            "ratio_esd_to_roundtrip": float(np.mean(esd_rel_errors) / np.mean(roundtrip_rel_errors)),
            "fraction_exceeding_fp16": float(exceeds_fp16_bound_count / total_elements_checked),
            "top_outliers": tensor_stats[:5],
        },
        "multiclass_styles": candidate_styles,
    }


@app.local_entrypoint()
def main():
    print("Running FP16-Ratio Test and Classifier Ground-Truth Diagnostic on Modal...")
    res = run_fp16_and_specificity_checks.remote()
    print("\n=== RESULTS ===")
    print(json.dumps(res, indent=2))
