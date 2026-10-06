"""Modal script to train, fetch, and lock all erased models into a persistent volume.

Models prepared and locked:
1. Clean ESD-x: trained for 300 steps directly from CompVis/stable-diffusion-v1-4 (seed 42).
2. Official ESD-x: downloaded from https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt.
3. UCE: closed-form Unified Concept Editing for 'Van Gogh' on SD 1.4.

All models are saved into Modal Volume 'heretic-models' and recorded in checkpoints.lock.yaml.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlretrieve

import modal

app = modal.App("heretic-prepare-models")

# Persistent Modal Volume for all models
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
    )
    .add_local_python_source("heretic_dit")
)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/models": models_volume},
    timeout=1800,
)
def prepare_models_on_modal() -> Dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from safetensors.torch import save_file, load_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Preparing Erased Checkpoints on {torch.cuda.get_device_name(0)} ===")
    models_dir = Path("/root/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. Download Official ESD-x Checkpoint
    # -------------------------------------------------------------------------
    official_esd_url = "https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt"
    official_esd_path = models_dir / "official_esd_sd14_van_gogh.pt"
    if not official_esd_path.exists():
        print(f"Downloading official ESD-x checkpoint from {official_esd_url}...")
        urlretrieve(official_esd_url, official_esd_path)
        print(f"Downloaded official ESD-x ({official_esd_path.stat().st_size / (1024*1024):.1f} MB).")
    else:
        print(f"Official ESD-x already present at {official_esd_path}.")

    with official_esd_path.open("rb") as f:
        official_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"Official ESD-x SHA-256: {official_sha}")
    results["official_esd_sha"] = official_sha

    # -------------------------------------------------------------------------
    # 2. Train Clean ESD-x Control Checkpoint (300 steps, lr=5e-5, seed 42)
    # -------------------------------------------------------------------------
    clean_esd_path = models_dir / "clean_esd_x_sd14_van_gogh.safetensors"
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

    if not clean_esd_path.exists():
        print("\nTraining Clean ESD-x Control Checkpoint (300 steps, seed 42)...")
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

        print("Pre-sampling 16 clean reference trajectories...")
        precomputed_trajectories = []
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

        trainable_params = []
        for name, param in unet.named_parameters():
            if "attn2.to_k" in name or "attn2.to_v" in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-2)
        t_start = time.perf_counter()
        for step in range(300):
            optimizer.zero_grad()
            sample = random.choice(precomputed_trajectories)
            t_val = torch.tensor([sample["t"]], device=device)
            latents_val = sample["latents"]
            text_emb_val = sample["text_emb"]
            e_0 = sample["pred_neg"]
            e_p = sample["pred_pos"]
            target = e_0 - 1.0 * (e_p - e_0)
            pred_new = unet(latents_val, t_val, encoder_hidden_states=text_emb_val).sample
            loss = F.mse_loss(pred_new, target)
            loss.backward()
            optimizer.step()

        esd_time = time.perf_counter() - t_start
        print(f"Clean ESD-x trained in {esd_time:.2f}s.")

        # Verify bit-identity on all non-cross-attn weights
        clean_state = {k: v.cpu().clone() for k, v in unet.state_dict().items()}
        non_cross_diffs = []
        for k in base_unet_state:
            if "attn2.to_k" not in k and "attn2.to_v" not in k:
                diff = (base_unet_state[k] - clean_state[k]).abs().max().item()
                non_cross_diffs.append(diff)
        assert max(non_cross_diffs) == 0.0, f"Bit identity failed: max diff = {max(non_cross_diffs)}"
        print(f"Bit-identity verified: {len(non_cross_diffs)} non-cross-attn tensors are bit-identical.")

        # Save to safetensors
        save_file(clean_state, str(clean_esd_path))
        print(f"Saved Clean ESD-x to {clean_esd_path}.")
    else:
        print(f"Clean ESD-x already exists at {clean_esd_path}.")

    with clean_esd_path.open("rb") as f:
        clean_esd_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"Clean ESD-x SHA-256: {clean_esd_sha}")
    results["clean_esd_sha"] = clean_esd_sha

    # -------------------------------------------------------------------------
    # 3. Compute UCE Checkpoint (Closed-Form Unified Concept Editing)
    # -------------------------------------------------------------------------
    uce_path = models_dir / "uce_sd14_van_gogh.safetensors"
    if not uce_path.exists():
        print("\nComputing UCE (Unified Concept Editing) closed-form checkpoint on SD 1.4...")
        pipe.unet.load_state_dict(base_unet_state)

        # UCE concept definitions
        concept_to_erase = "a painting in the style of Van Gogh"
        guided_concept = "a painting in the style of art"
        preserve_concepts = [
            "a painting in the style of Claude Monet",
            "a painting in the style of Rembrandt",
            "a painting in the style of Pablo Picasso",
            "a painting in the style of Andy Warhol",
            "a photograph of an apple on a table",
            "a photograph of a sports car",
            "a photograph of a dog playing in a park",
            "a landscape photograph of mountains",
        ]

        def get_text_emb(text: str) -> torch.Tensor:
            tok = tokenizer([text], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
            return text_encoder(tok)[0]  # [1, 77, 768]

        with torch.no_grad():
            c_e = get_text_emb(concept_to_erase).mean(dim=1).squeeze(0)  # [768]
            c_guide = get_text_emb(guided_concept).mean(dim=1).squeeze(0)  # [768]
            c_p_list = [get_text_emb(p).mean(dim=1).squeeze(0) for p in preserve_concepts]
            C_p = torch.stack(c_p_list, dim=0)  # [M, 768]

        # Regularized covariance of preserve concepts
        # Cov = (C_p^T C_p) / M + lambda * I
        lambda_reg = 0.1
        M = C_p.shape[0]
        cov_preserve = (C_p.T @ C_p) / M + lambda_reg * torch.eye(768, device=device)
        cov_inv = torch.linalg.inv(cov_preserve)

        # UCE edits cross-attention to_v projection weights
        uce_unet_state = copy.deepcopy(base_unet_state)
        for name, param in pipe.unet.named_parameters():
            if "attn2.to_v.weight" in name:
                w_0 = param.data.float()  # [d_out, 768]
                # Target output vector v* = w_0 @ c_guide
                v_star = w_0 @ c_guide  # [d_out]
                w_c_e = w_0 @ c_e       # [d_out]
                delta_v = v_star - w_c_e  # [d_out]
                # Closed form rank-1 update: delta_W = delta_v * (cov_inv @ c_e)^T / (c_e^T @ cov_inv @ c_e)
                inv_c_e = cov_inv @ c_e  # [768]
                denom = (c_e @ inv_c_e).item()
                delta_w = torch.outer(delta_v, inv_c_e) / denom
                uce_unet_state[name] = (w_0 + delta_w).cpu()

        save_file(uce_unet_state, str(uce_path))
        print(f"Saved UCE checkpoint to {uce_path}.")
    else:
        print(f"UCE checkpoint already exists at {uce_path}.")

    with uce_path.open("rb") as f:
        uce_sha = hashlib.sha256(f.read()).hexdigest()
    print(f"UCE SHA-256: {uce_sha}")
    results["uce_sha"] = uce_sha

    # Commit volume changes
    models_volume.commit()
    print("Modal Volume 'heretic-models' committed successfully.")

    return results


@app.local_entrypoint()
def main():
    print("Launching Model Preparation and Locking on Modal...")
    res = prepare_models_on_modal.remote()
    print("Results:", json.dumps(res, indent=2))

    # Write to local lockfile
    from heretic_dit.benchmarks.checkpoints import CheckpointEntry, write_lockfile

    entries = [
        CheckpointEntry(
            method="esd",
            base_model="sd14",
            concept="van-gogh",
            source="modal:heretic-models/clean_esd_x_sd14_van_gogh.safetensors",
            origin="trained",
            files={"clean_esd_x_sd14_van_gogh.safetensors": res["clean_esd_sha"]},
            seed=42,
            notes="Clean ESD-x control (300 steps, lr=5e-5, Delta W non-cross-attn == 0.0).",
        ),
        CheckpointEntry(
            method="esd-official",
            base_model="sd14",
            concept="van-gogh",
            source="https://erasing.baulab.info/weights/esd_models/art/diffusers-VanGogh-ESDx1-UNET.pt",
            origin="released",
            files={"official_esd_sd14_van_gogh.pt": res["official_esd_sha"]},
            notes="Official ICCV 2023 released ESD style-erasure checkpoint for Van Gogh.",
        ),
        CheckpointEntry(
            method="uce",
            base_model="sd14",
            concept="van-gogh",
            source="modal:heretic-models/uce_sd14_van_gogh.safetensors",
            origin="trained",
            files={"uce_sd14_van_gogh.safetensors": res["uce_sha"]},
            seed=42,
            notes="Unified Concept Editing (UCE WACV 2024) closed-form projection on SD 1.4.",
        ),
    ]

    lock_path = write_lockfile("checkpoints.lock.yaml", entries)
    print(f"Successfully locked provenance into {lock_path}!")


if __name__ == "__main__":
    main()
