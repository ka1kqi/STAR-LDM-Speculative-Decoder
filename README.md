# Hybrid Diffusion–Speculative Autoregressive LM

A research implementation of STAR-LDM-style **Stop–Think–AutoRegress** architecture
with **Qwen2.5-0.5B** as the AR backbone and **Sentence-T5** continuous structure
embeddings as the plan representation.

## Architecture overview

```
Prefix tokens ──► AR Backbone (Qwen2.5-0.5B) ──► prefix hidden states
                                                        │
                                           ┌────────────▼──────────────┐
                                           │  DiT1 (PromptEncoder)     │
                                           │  + noisy plan z_t + time  │
                                           └────────────┬──────────────┘
                                                        │ h1
                                           ┌────────────▼──────────────┐
                                           │  DiT2 (DiffusionPred.)    │
                                           │  → v-prediction → z_hat   │
                                           └────────────┬──────────────┘
                                                        │ z_hat
                                           ┌────────────▼──────────────┐
                                           │  Soft-Prompt Projector    │
                                           │  → K=8 soft prompt tokens │
                                           └────────────┬──────────────┘
                                                        │
                                           ┌────────────▼──────────────┐
                                           │  AR Backbone (generate)   │
                                           │  [soft_prompt | tokens]   │
                                           └───────────────────────────┘
```

**Plan embedding:** Structure features are extracted from continuation text,
canonically serialized, and encoded by Sentence-T5 (768-dim).

**Speculative decoding (optional):** Lossless (distribution-preserving) or
approximate (logSNR-adaptive threshold) modes using a draft AR model.

## Quick start

### 1. Create environment

```bash
conda create -n diffspec python=3.11 -y
conda activate diffspec
pip install -e .
pip install -r requirements.txt
```

### 2. Run unit tests (no GPU, no model download)

```bash
pytest -q -m "not slow"
```

### 3. Run smoke test (downloads Qwen2.5-0.5B ~1GB first time)

```bash
pytest -q -m slow
```

### 4. Preprocess data

```bash
# Full FineWeb preprocessing (streaming, caches Sentence-T5 embeddings)
python scripts/preprocess_data.py --config configs/default.yaml --max-samples 10000

# Toy mode (no network required for data, but needs Sentence-T5 model)
# Training script handles toy data internally — skip this step for toy runs
```

### 5. Train hybrid model

```bash
# Toy smoke test (CPU, ~2 minutes)
python scripts/train_hybrid.py --config configs/toy.yaml

# Full training (H100/A100)
python scripts/train_hybrid.py --config configs/default.yaml

# Multi-GPU with accelerate
accelerate launch scripts/train_hybrid.py --config configs/default.yaml

# Resume from checkpoint
python scripts/train_hybrid.py --config configs/default.yaml \
    --resume checkpoints/hybrid/step_50000
```

### 6. Train guidance MLP classifier

```bash
python scripts/train_guidance_mlp.py --config configs/default.yaml
```

### 7. Train draft model (for speculative decoding)

```bash
python scripts/train_draft_model.py --config configs/default.yaml
```

### 8. Generate text

```bash
python scripts/generate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --prompt "Once upon a time" \
    --num-samples 5

# With classifier guidance
python scripts/generate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --guidance-checkpoint checkpoints/guidance_mlp/step_50000 \
    --prompt "Once upon a time"

# With speculative decoding (set spec_decode.mode in config)
python scripts/generate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --draft-checkpoint checkpoints/draft/step_50000 \
    --prompt "Once upon a time"
```

### 9. Run C4 evaluation

```bash
python scripts/eval_c4.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --output results/c4_eval.json
```

Produces: MAUVE, generative perplexity (Llama-3.2-3B), lexical diversity.

### 10. Run StoryCloze LLM-as-judge evaluation

```bash
# Requires ANTHROPIC_API_KEY (or OPENAI_API_KEY with backend=openai in config)
export ANTHROPIC_API_KEY=sk-ant-...

python scripts/eval_storycloze_judge.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --output results/storycloze_eval.json
```

Produces: win rate with bootstrap confidence intervals.

### 11. Generate evaluation plots

Plots are generated from JSON result files:

```python
from src.eval.plots import make_eval_plots
make_eval_plots("results/c4_eval.json", output_dir="plots", prefix="c4")
```

### 12. Benchmark throughput

```bash
python scripts/bench_throughput.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --num-samples 50

# With speculative decoding
python scripts/bench_throughput.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --draft-checkpoint checkpoints/draft/step_50000 \
    --num-samples 50
```

### 13. Profile inference breakdown

```bash
python scripts/profile_inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/hybrid/step_100000 \
    --num-runs 20
```

## Configuration

All hyperparameters are in YAML configs under `configs/`.

| Config | Purpose |
|--------|---------|
| `configs/default.yaml` | Full training with STAR-LDM defaults |
| `configs/toy.yaml` | Smoke-test on CPU in minutes |
| `configs/l4_small_batch.yaml` | Overlay for L4 / small GPUs |

To combine configs:

```python
from src.utils.config import load_config, merge_configs
base = load_config("configs/default.yaml")
override = load_config("configs/l4_small_batch.yaml")
cfg = merge_configs(base, override)
```

## STAR-LDM defaults implemented

From Tables 3 and 4 of the STAR-LDM paper:

| Parameter | Value | Source |
|-----------|-------|--------|
| Soft prompt length K | 8 | Table 3 |
| DiT layers | 6 | Table 3 |
| DiT hidden dim | 1024 | Table 3 |
| DiT heads | 16 | Table 3 |
| Head dim | 64 | Table 3 |
| FFN | SwiGLU | Table 3 |
| Normalization | Adaptive RMSNorm | Table 3 |
| Noise schedule | Cosine | Table 3 |
| Output parameterization | v-prediction | Table 3 |
| Loss weighting | sigmoid(logSNR) | Table 3 |
| Diffusion loss weight beta | 5.0 | Table 3 |
| Inference sampler | DDPM, 50 steps | Table 3 |
| Final noise sigma^2 | 0.1 | Table 3 |
| Decoding | nucleus p=0.95, rep penalty 1.2 | Paper |
| Guidance MLP hidden | 1536 | Table 4 |
| Guidance MLP blocks | 4 | Table 4 |
| Guidance MLP lr | 1e-4 | Table 4 |
| Guidance MLP batch | 256 | Table 4 |

Parameters marked **UNSPECIFIED** in configs are not specified in the paper
and use reasonable defaults.

## Checkpoint structure

```
checkpoints/
├── hybrid/step_{N}/
│   ├── model.pt
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── config.json
│   └── step.txt
├── guidance_mlp/step_{N}/
│   ├── model.pt
│   └── config.json
└── draft/step_{N}/
    ├── model.pt
    └── config.json
```

## Hardware requirements

| Task | Minimum GPU | Recommended |
|------|-------------|-------------|
| Training (full) | A100 80GB | H100 80GB |
| Training (toy) | CPU | Any |
| Evaluation | L4 24GB | A100 |
| Inference | L4 24GB | L4+ |

Use `configs/l4_small_batch.yaml` as an overlay for smaller GPUs.

## Project structure

```
diff_spec_AR_LM/
├── configs/                    # YAML configuration files
│   ├── default.yaml           # Full STAR-LDM defaults
│   ├── toy.yaml               # CPU smoke-test config
│   ├── l4_small_batch.yaml    # Small GPU overlay
│   └── storycloze_judge_template.txt
├── scripts/                    # Executable scripts
│   ├── preprocess_data.py
│   ├── train_hybrid.py
│   ├── train_guidance_mlp.py
│   ├── train_draft_model.py
│   ├── generate.py
│   ├── eval_c4.py
│   ├── eval_storycloze_judge.py
│   ├── bench_throughput.py
│   └── profile_inference.py
├── src/                        # Main package
│   ├── data/                  # Data loading, structure extraction, embeddings
│   │   ├── dataset.py
│   │   ├── embedding.py
│   │   └── structure.py
│   ├── models/                # DiT blocks, diffusion, hybrid model, samplers
│   │   ├── diffusion.py
│   │   ├── dit.py
│   │   ├── guidance_mlp.py
│   │   ├── hybrid.py
│   │   └── samplers.py
│   ├── spec_decode/           # Speculative decoding (lossless + approximate)
│   │   ├── lossless.py
│   │   └── approximate.py
│   ├── infer/                 # Unified inference pipeline
│   │   └── pipeline.py
│   ├── eval/                  # Evaluation harnesses + plotting
│   │   ├── c4_eval.py
│   │   ├── storycloze.py
│   │   └── plots.py
│   └── utils/                 # Config, logging, seeding, profiling
│       ├── config.py
│       ├── logging.py
│       ├── seed.py
│       └── profiling.py
├── tests/                      # Unit + smoke tests
├── requirements.txt
├── setup.py
├── pytest.ini
└── README.md
```
