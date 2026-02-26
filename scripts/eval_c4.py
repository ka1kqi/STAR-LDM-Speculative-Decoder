#!/usr/bin/env python3
"""Run C4 evaluation: MAUVE, generative perplexity, lexical diversity.

Usage:
    python scripts/eval_c4.py --config configs/default.yaml \
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
from src.eval.c4_eval import evaluate_c4
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("eval_c4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/c4_eval.json")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_cfg = cfg["model"]
    infer_cfg = cfg.get("inference", {})
    sampler_cfg = cfg.get("sampler", {})
    eval_cfg = cfg.get("eval", {}).get("c4", {})

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
        max_new_tokens=eval_cfg.get("cont_len", 64),
    )

    metrics = evaluate_c4(
        pipeline=pipeline,
        tokenizer=tokenizer,
        n_samples=eval_cfg.get("n_samples", 5000),
        prefix_len=eval_cfg.get("prefix_len", 32),
        cont_len=eval_cfg.get("cont_len", 64),
        ppl_model=eval_cfg.get("ppl_model", "meta-llama/Llama-3.2-3B"),
        device=str(device),
        compute_ppl=eval_cfg.get("compute_ppl", True),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
