#!/usr/bin/env python3
"""Benchmark generation throughput: tokens/sec, wall-clock per sample, spec-decode stats.

Usage:
    python scripts/bench_throughput.py --config configs/default.yaml \
        --checkpoint checkpoints/hybrid/step_100000 \
        --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.hybrid import HybridModel
from src.infer.pipeline import InferencePipeline
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.profiling import Timer

log = get_logger("bench_throughput")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--draft-checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/throughput.json")
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

    # Draft model
    draft_model = None
    if args.draft_checkpoint and spec_cfg.get("mode"):
        draft_model = AutoModelForCausalLM.from_pretrained(
            cfg["draft"]["model_name"], trust_remote_code=True
        ).to(device)
        draft_model.load_state_dict(
            torch.load(Path(args.draft_checkpoint) / "model.pt", map_location=device)
        )
        draft_model.eval()

    max_new = infer_cfg.get("max_new_tokens", 64)

    # Benchmark modes
    modes = [("standard", None)]
    if draft_model:
        if spec_cfg.get("mode") == "lossless":
            modes.append(("lossless", "lossless"))
        if spec_cfg.get("mode") == "approximate":
            modes.append(("approximate", "approximate"))
        # Also test both
        modes.append(("lossless", "lossless"))
        modes.append(("approximate", "approximate"))

    # Deduplicate
    seen = set()
    unique_modes = []
    for name, mode in modes:
        if name not in seen:
            seen.add(name)
            unique_modes.append((name, mode))
    modes = unique_modes

    results = {}

    for mode_name, spec_mode in modes:
        log.info(f"Benchmarking mode: {mode_name}")

        pipeline = InferencePipeline(
            hybrid_model=model,
            sampler_type=sampler_cfg.get("type", "ddpm"),
            sampler_steps=sampler_cfg.get("num_steps", 50),
            cfg_scale=infer_cfg.get("cfg_scale", 1.5),
            spec_mode=spec_mode,
            draft_model=draft_model if spec_mode else None,
            draft_k=spec_cfg.get("draft_k", 4),
            top_p=infer_cfg.get("top_p", 0.95),
            temperature=infer_cfg.get("temperature", 1.0),
            repetition_penalty=infer_cfg.get("repetition_penalty", 1.2),
            max_new_tokens=max_new,
        )

        timer = Timer()
        total_tokens = 0
        total_accepted = 0
        total_drafted = 0
        total_target_calls = 0

        prefix_text = "The quick brown fox jumps over the lazy dog."
        input_ids = tokenizer.encode(prefix_text, return_tensors="pt").to(device)

        # Warmup
        _ = pipeline(input_ids)

        for i in range(args.num_samples):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.start()
            result = pipeline(input_ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer.stop()

            gen_len = result["generated_ids"].shape[1] - input_ids.shape[1]
            total_tokens += gen_len

            if result["spec_stats"]:
                total_accepted += result["spec_stats"].get("n_accepted", 0)
                total_drafted += result["spec_stats"].get("n_drafted", 0)
                total_target_calls += result["spec_stats"].get("n_target_calls", 0)

        tokens_per_sec = total_tokens / timer.total if timer.total > 0 else 0
        wall_per_sample = timer.avg

        mode_results = {
            "tokens_per_sec": round(tokens_per_sec, 2),
            "wall_clock_per_sample_ms": round(wall_per_sample * 1000, 2),
            "total_tokens": total_tokens,
            "total_time_s": round(timer.total, 3),
            "n_samples": args.num_samples,
        }

        if total_drafted > 0:
            mode_results["acceptance_rate"] = round(total_accepted / total_drafted, 4)
            mode_results["avg_accepted_per_block"] = round(
                total_accepted / max(total_target_calls, 1), 2
            )
            mode_results["n_target_calls"] = total_target_calls

        results[mode_name] = mode_results
        log.info(f"  {mode_name}: {tokens_per_sec:.1f} tok/s, {wall_per_sample*1000:.1f} ms/sample")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
