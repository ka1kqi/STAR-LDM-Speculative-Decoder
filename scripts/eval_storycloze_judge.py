#!/usr/bin/env python3
"""Run StoryCloze LLM-as-judge evaluation.

Usage:
    python scripts/eval_storycloze_judge.py --config configs/default.yaml \
        --checkpoint checkpoints/hybrid/step_100000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.models.hybrid import HybridModel
from src.infer.pipeline import InferencePipeline
from src.eval.storycloze import evaluate_storycloze
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("eval_storycloze")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/storycloze_eval.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})
    sampler_cfg = cfg.get("sampler", {})
    sc_cfg = cfg.get("eval", {}).get("storycloze", {})

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

    pipeline = InferencePipeline(
        hybrid_model=model,
        sampler_type=sampler_cfg.get("type", "ddpm"),
        sampler_steps=sampler_cfg.get("num_steps", 50),
        cfg_scale=infer_cfg.get("cfg_scale", 1.5),
        top_p=infer_cfg.get("top_p", 0.95),
        temperature=infer_cfg.get("temperature", 1.0),
        repetition_penalty=infer_cfg.get("repetition_penalty", 1.2),
        max_new_tokens=infer_cfg.get("max_new_tokens", 64),
    )

    results = evaluate_storycloze(
        pipeline=pipeline,
        tokenizer=tokenizer,
        n_samples=sc_cfg.get("n_samples", 200),
        backend=sc_cfg.get("backend", "anthropic"),
        judge_model=sc_cfg.get("judge_model"),
        template_path=sc_cfg.get("template_path"),
        device=str(device),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove raw_judgments from JSON for cleaner output
    results_clean = {k: v for k, v in results.items() if k != "raw_judgments"}
    with open(out_path, "w") as f:
        json.dump(results_clean, f, indent=2)

    # Save full judgments separately
    with open(out_path.with_suffix(".judgments.json"), "w") as f:
        json.dump(results.get("raw_judgments", []), f, indent=2)

    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
