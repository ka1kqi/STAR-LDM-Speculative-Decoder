# Architecture Deep Dive

## The Two Phases: THINK then TALK

The model follows the STAR-LDM "Stop-Think-AutoRegress" pattern. There are two distinct phases at inference time: a **diffusion planning phase** (THINK) that produces a latent plan, and an **autoregressive generation phase** (TALK) that generates text conditioned on that plan.

---

## Phase 1: THINK — Diffusion Planning

The goal is to produce a **plan embedding** `z_hat` (a 768-dim vector) that encodes high-level structural intent (length, style, whether there's a list, etc.) for what the continuation should look like.

### Training the planner

During training, we have ground-truth plan embeddings `z0` — these come from running Sentence-T5 on a canonical serialization of structure features extracted from the actual continuation text. The diffusion model learns to *denoise* a corrupted version of `z0` back to `z0`.

Here's the context flow:

```
1. prefix_ids → AR backbone (Qwen2.5-0.5B) → prefix_hidden (B, Lp, 896)
                                                     │
2. Sample random timestep t ~ U[0,1]                 │
   Add noise to z0:  z_t = √ᾱ·z0 + √(1-ᾱ)·ε       │
                              │                       │
3.        ┌───────────────────┴───────────────────────┘
          │
          ▼
   DiT1 (PromptEncoder):
     - Projects prefix_hidden (896-dim) → 1024-dim
     - Projects z_t (768-dim) → 1024-dim, appends as extra token
     - Adds sinusoidal timestep embedding of t as conditioning
     - Runs 6 DiT blocks (self-attention + SwiGLU FFN + AdaptiveRMSNorm)
     - Output: h1 (B, Lp+1, 1024)
                    │
                    ▼
   DiT2 (DiffusionPrediction):
     - Projects z_t → 1024-dim, appends to h1 as extra token
     - Adds its own timestep embedding
     - Runs 6 more DiT blocks
     - Takes the last token's output, projects 1024 → 768
     - Output: v_pred (B, 768) — the v-prediction
```

**v-prediction**: Instead of predicting the noise directly, the model predicts `v = √ᾱ·ε - √(1-ᾱ)·z0`. This is numerically better-conditioned. The loss is MSE weighted by `sigmoid(logSNR(t))`, which down-weights very noisy and very clean timesteps.

### Inference (denoising)

At inference, we start from pure noise `z_T ~ N(0,I)` and iteratively denoise using DDPM (50 steps) or DPM-Solver++ (20 steps). Each step calls `denoise_fn` which runs prefix_hidden through DiT1 → DiT2 to get `v_pred`, then the sampler uses it to step toward the clean plan:

```
z_T (pure noise) → denoise_fn(z_49) → ... → denoise_fn(z_1) → z_hat
```

**Classifier-Free Guidance (CFG)**: During training, the prefix is randomly dropped (replaced with zeros) 10% of the time. At inference with `cfg_scale > 1`:

```
v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
```

This steers the plan toward being more consistent with the actual prefix.

**Classifier Guidance** (optional): A separate noise-conditioned MLP (`GuidanceMLP`) predicts structure attributes from `(z_t, t)`. Its gradient `∇_z log p(y|z_t, t)` is added to the v-prediction to steer toward desired attributes.

---

## Phase 2: TALK — Autoregressive Generation

Once we have `z_hat`, we convert it into **soft prompt tokens** that condition the AR backbone:

```
z_hat (768-dim)
     │
     ▼
DiT1 at t=0:  prefix_hidden + z_hat → h1 (B, Lp+1, 1024)
     │
     ▼
Mean-pool h1 → h1_pooled (B, 1024)
     │
     ▼
soft_prompt_proj:  Linear(1024→1024) → SiLU → Linear(1024 → 896*8)
     │
     ▼
Reshape → soft_prompt (B, 8, 896)  — 8 virtual tokens in LM embedding space
```

Then standard autoregressive generation with the AR backbone:

```
AR backbone receives: [prefix_embeds | soft_prompt | generated_so_far]
                       ─────────────   ──────────   ─────────────────
                       real tokens      8 virtual    tokens generated
                       from input       plan tokens  one at a time

First forward pass:
  inputs_embeds = [prefix_embeds, soft_prompt]  → logits, KV cache

Then token-by-token with KV cache:
  for each step:
    next_token_embed → AR backbone (with cached KVs) → next logits
    nucleus sample (top-p=0.95) with repetition penalty (1.2)
```

The soft prompt tokens are the bridge — they inject the plan's structural intent into the AR model's context without modifying its weights.

---

## Training Loss

All three components train jointly:

```
L_total = L_LM + β·L_DM + γ·L_align

L_LM   = standard AR cross-entropy on prefix (next-token)
        + cross-entropy on continuation conditioned on soft prompt
L_DM   = sigmoid(logSNR)-weighted MSE on v-prediction
L_align = optional structure-attribute classification from LM hidden states
```

With `β=5.0` (from the paper), the diffusion loss is weighted heavily to ensure the planner learns a good representation.

---

## Speculative Decoding (optional)

A separate draft AR model generates `k=4` candidate tokens at a time, then the target model verifies them in a single forward pass. Two modes:

- **Lossless**: Classic rejection sampling — `accept if p_target/p_draft > rand()`. Preserves exact target distribution.
- **Approximate**: Accept if `p_target(token) >= τ(c)` where `τ` adapts to plan confidence via `τ = τ_min + (τ_max - τ_min)·σ(c·logSNR)`. High-confidence plans → higher threshold → more aggressive speculation.

---

## Dtype Flow

Everything runs in **bf16** to match the Qwen2.5-0.5B backbone, except diffusion schedule math (`cosine_alpha_bar`, `logSNR`, `q_sample`, `v_target`) which stays in **fp32** for numerical stability. Cast points are at the boundaries: fp32 schedule outputs → bf16 before entering DiT, and `v_pred` → fp32 before computing MSE loss.

---

## Key Source Files

| File | Role |
|------|------|
| `src/models/hybrid.py` | `HybridModel` — orchestrates everything |
| `src/models/dit.py` | DiT blocks, `PromptEncoder` (DiT1), `DiffusionPrediction` (DiT2) |
| `src/models/diffusion.py` | Cosine schedule, logSNR, v-prediction math |
| `src/models/samplers.py` | DDPM and DPM-Solver++(2M) samplers |
| `src/models/guidance_mlp.py` | Noise-conditioned classifier for guidance |
| `src/spec_decode/lossless.py` | Lossless speculative decoding |
| `src/spec_decode/approximate.py` | Approximate speculative decoding with adaptive τ |
| `src/infer/pipeline.py` | Unified inference pipeline tying it all together |
| `src/data/structure.py` | Structure feature extraction + canonical serialization |
| `src/data/embedding.py` | Sentence-T5 encoder with disk cache |
