#!/usr/bin/env python3
"""Generate text using the trained hybrid model.

Usage:
    python scripts/generate.py --config configs/default.yaml \
        --checkpoint checkpoints/hybrid/step_100000 \
        --prompt "The quick brown fox"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.models.hybrid import HybridModel
from src.models.guidance_mlp import GuidanceMLP
from src.models.samplers import DDPMSampler, DPMSolverPPSampler
from src.models.diffusion import DiffusionSchedule
from src.infer.pipeline import InferencePipeline
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("generate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--guidance-checkpoint", type=str, default=None)
    parser.add_argument("--draft-checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})
    sampler_cfg = cfg.get("sampler", {})
    spec_cfg = cfg.get("spec_decode", {})

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load hybrid model
    model = HybridModel(
        ar_model_name=model_cfg["ar_model_name"],
        d_plan=model_cfg.get("d_plan", 768),
        d_dit=model_cfg.get("d_dit", 1024),
        dit_layers=model_cfg.get("dit_layers", 6),
        dit_heads=model_cfg.get("dit_heads", 16),
        dit_head_dim=model_cfg.get("dit_head_dim", 64),
        soft_prompt_len=model_cfg.get("soft_prompt_len", 8),
        beta=model_cfg.get("beta", 5.0),
        gamma=model_cfg.get("gamma", 0.0),
        final_noise_sigma2=model_cfg.get("final_noise_sigma2", 0.1),
    ).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()

    # Optional guidance MLP
    guidance_mlp = None
    if args.guidance_checkpoint:
        guidance_mlp = GuidanceMLP(
            d_plan=cfg["guidance_mlp"].get("d_plan", 768),
            d_hidden=cfg["guidance_mlp"].get("d_hidden", 1536),
            n_blocks=cfg["guidance_mlp"].get("n_blocks", 4),
        ).to(device)
        guidance_mlp.load_state_dict(
            torch.load(Path(args.guidance_checkpoint) / "model.pt", map_location=device)
        )
        guidance_mlp.eval()

    # Optional draft model
    draft_model = None
    if args.draft_checkpoint and spec_cfg.get("mode"):
        from transformers import AutoModelForCausalLM
        draft_model = AutoModelForCausalLM.from_pretrained(
            cfg["draft"]["model_name"], trust_remote_code=True
        ).to(device)
        draft_model.load_state_dict(
            torch.load(Path(args.draft_checkpoint) / "model.pt", map_location=device)
        )
        draft_model.eval()

    pipeline = InferencePipeline(
        hybrid_model=model,
        sampler_type=sampler_cfg.get("type", "ddpm"),
        sampler_steps=sampler_cfg.get("num_steps", 50),
        cfg_scale=infer_cfg.get("cfg_scale", 1.5),
        guidance_mlp=guidance_mlp,
        guidance_scale=infer_cfg.get("guidance_scale", 0.0),
        spec_mode=spec_cfg.get("mode"),
        draft_model=draft_model,
        draft_k=spec_cfg.get("draft_k", 4),
        top_p=infer_cfg.get("top_p", 0.95),
        temperature=infer_cfg.get("temperature", 1.0),
        repetition_penalty=infer_cfg.get("repetition_penalty", 1.2),
        max_new_tokens=infer_cfg.get("max_new_tokens", 64),
    )

    for i in range(args.num_samples):
        input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
        result = pipeline(input_ids)
        text = tokenizer.decode(result["generated_ids"][0], skip_special_tokens=True)
        print(f"\n=== Sample {i+1} ===")
        print(text)
        if result["spec_stats"]:
            print(f"Spec stats: {result['spec_stats']}")


if __name__ == "__main__":
    main()
