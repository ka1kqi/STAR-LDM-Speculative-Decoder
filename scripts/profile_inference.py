#!/usr/bin/env python3
"""Profile inference: break down time spent in planning vs. generation.

Usage:
    python scripts/profile_inference.py --config configs/default.yaml \
        --checkpoint checkpoints/hybrid/step_100000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.models.hybrid import HybridModel
from src.models.samplers import DDPMSampler, DPMSolverPPSampler
from src.models.diffusion import DiffusionSchedule
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.profiling import Timer, profile_cuda

log = get_logger("profile")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/profile.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})
    sampler_cfg = cfg.get("sampler", {})

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = HybridModel(
        ar_model_name=model_cfg["ar_model_name"],
        d_plan=model_cfg.get("d_plan", 768),
        d_dit=model_cfg.get("d_dit", 1024),
        dit_layers=model_cfg.get("dit_layers", 6),
        dit_heads=model_cfg.get("dit_heads", 16),
        dit_head_dim=model_cfg.get("dit_head_dim", 64),
        soft_prompt_len=model_cfg.get("soft_prompt_len", 8),
        final_noise_sigma2=model_cfg.get("final_noise_sigma2", 0.1),
    ).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint) / "model.pt", map_location=device))
    model.eval()

    schedule = DiffusionSchedule()
    if sampler_cfg.get("type") == "dpm_solver_pp":
        sampler = DPMSolverPPSampler(schedule, num_steps=sampler_cfg.get("dpm_solver_steps", 20))
    else:
        sampler = DDPMSampler(schedule, num_steps=sampler_cfg.get("num_steps", 50))

    prefix_text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(prefix_text, return_tensors="pt").to(device)

    plan_timer = Timer()
    gen_timer = Timer()

    # Warmup
    z_hat = model.plan(input_ids, sampler, cfg_scale=infer_cfg.get("cfg_scale", 1.5))
    _ = model.generate_from_plan(input_ids, z_hat, max_new_tokens=infer_cfg.get("max_new_tokens", 64))

    for _ in range(args.num_runs):
        # Profile planning
        plan_timer.start()
        z_hat = model.plan(input_ids, sampler, cfg_scale=infer_cfg.get("cfg_scale", 1.5))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        plan_timer.stop()

        # Profile generation
        gen_timer.start()
        _ = model.generate_from_plan(
            input_ids, z_hat,
            max_new_tokens=infer_cfg.get("max_new_tokens", 64),
            top_p=infer_cfg.get("top_p", 0.95),
            repetition_penalty=infer_cfg.get("repetition_penalty", 1.2),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_timer.stop()

    results = {
        "plan_avg_ms": round(plan_timer.avg * 1000, 2),
        "plan_total_ms": round(plan_timer.total * 1000, 2),
        "gen_avg_ms": round(gen_timer.avg * 1000, 2),
        "gen_total_ms": round(gen_timer.total * 1000, 2),
        "total_avg_ms": round((plan_timer.avg + gen_timer.avg) * 1000, 2),
        "plan_fraction": round(plan_timer.avg / (plan_timer.avg + gen_timer.avg + 1e-10), 3),
        "gen_fraction": round(gen_timer.avg / (plan_timer.avg + gen_timer.avg + 1e-10), 3),
        "num_runs": args.num_runs,
        "sampler": sampler_cfg.get("type", "ddpm"),
        "sampler_steps": sampler_cfg.get("num_steps", 50),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Planning: {results['plan_avg_ms']:.1f} ms avg ({results['plan_fraction']*100:.1f}%)")
    log.info(f"Generation: {results['gen_avg_ms']:.1f} ms avg ({results['gen_fraction']*100:.1f}%)")
    log.info(f"Total: {results['total_avg_ms']:.1f} ms avg")
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
