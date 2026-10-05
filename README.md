# Heretic-DiT ⚡

**Reproducible Subspace Abliteration and Concept Erasure for Diffusion Transformers (DiTs) and Non-Autoregressive Models.**

## Overview
Heretic-DiT addresses the core limitations of autoregressive abliteration tools when applied to diffusion and non-autoregressive architectures (e.g. DiffusionGemma, Stable Diffusion, Flux, Lumina).

### Key Features
* **Closed-Form Analytic Projection**: Deterministic null-space and covariance-regularized projectors (MACE / CURE style) with zero Optuna/L-BFGS optimization jitter.
* **First-Class MoE Support**: Batched tensor operations (`torch.bmm`) across 3D expert weight tensors (e.g. `[128, D_out, D_in]`).
* **Multi-Timestep Trajectory Probing**: Measures activation shifts along actual reverse diffusion trajectories ($x_T \to x_0$) rather than static zero-canvas forward passes.
* **Tied Memory Safety**: Guarantees consistency across tied encoder/decoder weights (`data_ptr`).
* **Deterministic Noise Schedules**: Trajectory-level KL divergence with pinned RNG noise seeds.
