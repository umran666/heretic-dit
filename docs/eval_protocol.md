# Evaluation Protocol

This document specifies exactly what Heretic-DiT measures, how, and which
numbers go in the paper. Everything here is implemented in `heretic_dit/eval/`,
`heretic_dit/baselines/`, `heretic_dit/benchmarks/`, `heretic_dit/reporting/`,
and driven by `scripts/run_experiment.py`.

## 0. Scope (hard constraint)

Only **benign proxy concepts** are used: the 10 Imagenette classes, a small
list of artist styles (e.g. Van Gogh), and celebrity-erasure targets of the
kind used in the ESD and MACE papers. No NSFW prompts, datasets, concept
lists, or NSFW-oriented baseline configs exist anywhere in this repo. No
adversarial-prompt attack baselines (Ring-A-Bell, UnlearnDiffAtk,
MMA-Diffusion) are implemented; their released setups target unsafe concepts
and are out of scope. The concept list is an injectable config
(`configs/concepts_default.yaml`, override with `--concepts-config`).

## 1. Benchmark data (`heretic_dit/benchmarks/`)

- **Registry**: each concept has a name, category (`object` / `style` /
  `celebrity`), ≥10 prompt templates instantiated per concept, the shared
  neutral/control prompt set (40 concept-free prompts), and metadata
  (e.g. ImageNet wnid for Imagenette classes).
- **Split**: prompts are deterministically divided into a **search** split
  (usable by the Optuna proxy search) and a **held-out** split (never seen
  during search). The split is seeded from `seed` + concept slug via a
  platform-independent splitmix64 shuffle, so it is identical across machines
  and Python versions. `load_or_create_split` persists it to
  `splits/split_manifest.json` with a SHA-256 fingerprint; later runs reuse
  the identical split, and any config-vs-disk mismatch is a hard error.
- **List sizes** are configurable per category (`max_per_category`).

## 2. Ground-truth recovery (`heretic_dit/eval/validator.py`)

`DeterministicGenerativeValidator` implements the `GenerativeValidator`
protocol from `heretic_dit/interfaces.py`:

    validate(model, concept, num_samples, seed) -> RecoveryResult

- **Model access**: `model` is anything satisfying the local
  `GenerationAdapter` protocol (`eval/adapters.py`): one call per image,
  `generate(prompt, seed, steps, guidance) -> (C,H,W) float [0,1]`, plus a
  stable `model_id`. `CallableAdapter` wraps a function; `DiffusersAdapter`
  wraps a diffusers pipeline (Antigravity's pipeline hooks expose their
  SD 1.5 / SDXL / DiT models through one of these).
- **Seeds**: per-image seeds derive deterministically from
  `(model_id, prompt, run_seed, steps, guidance)` (SHA-256). Base and edited
  models therefore use different but reproducible seed streams, and adding a
  prompt never shifts other images' seeds.
- **Prompts**: recovery is measured **only on the held-out split**; when
  `num_samples` exceeds the held-out set, prompts cycle with per-index seeds.
- **Scoring**: a `ConceptClassifier` (protocol in `interfaces.py`) returns
  per-image confidence in [0,1]; recovery rate = fraction with confidence ≥
  threshold (default 0.5). Drift is the classifier's false-positive rate on
  the neutral prompts.
- **Batching / resumability**: classifier calls are batched
  (`ValidatorConfig.batch_size`); generated images are cached to
  `cache_dir/<sha256>.pt`, so interrupted runs resume exactly (a resumed run
  regenerates nothing a fresh run would have kept).

### Classifiers (`eval/classifiers.py`)

| Classifier | Use | Dependency |
|---|---|---|
| `ImagenetteClassifier` | object concepts; ImageNet-pretrained torchvision model reduced to the 10 Imagenette classes (softmax mass on the concept's wnid) | torchvision |
| `ClipZeroShotClassifier` | any concept; softmax over registered concepts' CLIP text embeddings | open_clip_torch |
| `ClipStyleScorer` | style concepts; style prompts vs. photographic counter-prompts | open_clip_torch |

All are lazy imports with actionable errors; tests use fakes satisfying the
same protocol.

**Erased-vs-unerased gap**: `recovery_gap(base_result, edited_result)` —
recovery of the unerased reference minus recovery of the edited model, on the
same prompts and seed scheme.

## 3. Quality metrics (`eval/quality.py`) — paired by construction

All comparisons are generated with **the same prompts and the same run-level
seed** for base and edited models (`generate_paired_images` /
`QualityEvaluator`), so deltas are not confounded by sampling noise:

- **FID**: Fréchet distance between Gaussians fit to features from a shared
  extractor. Default extractor is InceptionV3 (torchvision); a deterministic
  `RandomProjectionFeatures` stand-in exists for CPU tests. Features from
  *both* sets go through the identical extractor.
- **LPIPS**: elementwise-paired between same-index images (same seeds),
  mean over the set.
- **CLIP score**: `2.5 · max(cos, 0)` between each image and its prompt,
  averaged; reported for base and edited so the delta is visible.

## 4. Proxy validity (`eval/proxy_validity.py`) — the paper's sanity check

The Optuna search ranks edits by fast proxy metrics (prediction drift on
cached noisy latents; owned by `heretic_dit/metrics/`). To prove the proxy
can be trusted:

1. Sample trials across the proxy-value range the search actually visits
   (not only winners).
2. For each trial, compute real held-out recovery and real paired quality
   loss (`ProxyTrial`: `proxy_recovery`, `proxy_drift`, `real_recovery`,
   `real_quality_loss`).
3. `evaluate_proxy_validity(trials)` reports Spearman ρ (+ p-values) for:
   - proxy recovery vs. real recovery,
   - proxy drift vs. real quality loss,
   - composite utility (`w_r·recovery − w_d·drift/loss`) vs. real utility.

**Paper convention**: the proxy is reported trustworthy iff
ρ_recovery ≥ 0.7 and ρ_utility ≥ 0.7. The proxy-vs-real correlation figure
annotates both scatter panels with these ρ values. Constant/short inputs
raise errors rather than producing NaNs.

## 5. Baselines (`heretic_dit/baselines/`) — what the edit must beat

All implement the `BaselineRunner` protocol:
`run(erased_model, concept, budget) -> RecoveryResult` with a standardized
cost block `{wall_clock_sec, peak_vram_mb, trainable_params, sample_count}`
plus method-specific extras (`steps`, `lr`, `training_images`, ...).

| Baseline | Description | Key budget keys |
|---|---|---|
| `no-edit` | erased model as-is (lower bound) | validator, adapter_factory |
| `random-projection` | random orthonormal subspace with **matched alpha** (isolates the found subspace) | + apply_edit, model_dim |
| `unerased-reference` | base model (upper bound) | validator, adapter_factory |
| `textual-inversion` | pseudo-token embedding, few-shot, following Pham et al. (2023); real loop in `DiffusersTextualInversionBackend` | + steps, lr, training_prompts, backend |
| `lora` / `full` finetune (`FinetuneRecovery`) | few-step recovery on few-shot images; `steps` is the sweep parameter; peft-based LoRA in `DiffusersLoRABackend` | + steps, lr, rank, backend |

Real training backends are lazy imports; tests use in-repo dummy backends, so
the suite stays CPU-only and download-free.

## 6. Checkpoints (`scripts/prepare_erased_checkpoints.py`)

Publicly released erased checkpoints are downloaded where they exist
(ESD: Hugging Face `rohitgandikota/erasing-models`); where no verified release
exists (UCE, MACE artifacts in the manifest) the script refuses unless
`--train` is passed, then clones the official repo, resolves/pins a commit,
and runs the official training command with `--seed`. Every artifact lands in
`checkpoints.lock.yaml`: source, revision/commit, seed (trained only), exact
file hashes. `--verify` re-hashes everything; any mismatch is a hard error —
checkpoints are never silently substituted.

## 7. Experiments and reporting

`scripts/run_experiment.py` expands (erasure method × concept × base model ×
method/baseline × seed) into run specs, each with a content-derived `run_id`,
and writes `results/<run_id>/result.json`. Re-invocation skips runs whose
result matches the current config hash (resumable). Per-run execution is a
pluggable runner (`module:function` → `RecoveryResult`) wired by the repo
integration layer to real pipelines.

`heretic_dit/reporting/` produces:

- **Main table** (`main_table_latex`): recovery vs. drift/FID/LPIPS vs. cost
  per method, sectioned by erasure method.
- **Pareto figure** (`plot_pareto`): quality loss vs. recovery, main-method
  frontier as a step line, baselines overlaid.
- **Proxy-vs-real figure** (`plot_proxy_correlation`): both scatter panels
  with Spearman ρ annotations.
- **Per-erasure-method breakdown** (`breakdown_table_latex`,
  `plot_per_erasure_breakdown`): erasure methods fail differently, so no
  number in the paper averages over them.

All figures are written as PNG + PDF; tables as LaTeX (booktabs) and Markdown.

## 8. Which numbers go in the paper

1. **Recovery table**: held-out recovery rate per (erasure method × concept
   category × method), with the erased-vs-unerased gap against the
   `no-edit` / `unerased-reference` anchors.
2. **Quality table**: paired FID, LPIPS, CLIP-score delta on neutral prompts.
3. **Cost table**: wall-clock, peak VRAM, trainable params, images needed.
4. **Pareto figure**: recovery vs. quality loss with baselines overlaid.
5. **Proxy-validity figure**: Spearman ρ panels (this is the load-bearing
   sanity check for the search; report it even if favorable).

## 9. Reproducibility rules

- Fixed seeds: concepts config seed (splits), per-run seed (grid), per-image
  derived seeds, training seeds (backends), `RandomProjectionFeatures` seed.
- Prompt splits, checkpoint hashes, and run results are persisted artifacts;
  mismatches against persisted state are errors, never silent regeneration.
- Tests must run on CPU with no downloads (tiny stand-ins only).
