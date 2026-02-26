#!/usr/bin/env python3
"""Train the noise-conditioned guidance MLP classifier.

Trains on (z_t, t) → structure attribute labels.
Uses frozen Sentence-T5 embeddings + diffusion forward process to produce z_t.

Usage:
    python scripts/train_guidance_mlp.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.guidance_mlp import GuidanceMLP
from src.models.diffusion import DiffusionSchedule
from src.data.dataset import ToyDataset, FineWebDataset, build_dataloader
from src.data.embedding import SentenceT5Encoder
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.tracking import init_tracking, log_metrics, finish as finish_tracking
from transformers import AutoTokenizer

log = get_logger("train_guidance_mlp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    g_cfg = cfg.get("guidance_mlp", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    set_seed(cfg.get("training", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    if data_cfg.get("dataset_name") == "toy":
        dataset = ToyDataset(tokenizer, prefix_len=data_cfg.get("prefix_len", 32),
                             cont_len=data_cfg.get("cont_len", 64))
    else:
        st5 = SentenceT5Encoder(
            model_name=data_cfg.get("st5_model", "sentence-transformers/sentence-t5-xl"),
            cache_dir=data_cfg.get("st5_cache_dir", "cache/st5_embeddings"),
            device="cpu",
        )
        dataset = FineWebDataset(
            tokenizer=tokenizer,
            prefix_len=data_cfg.get("prefix_len", 32),
            cont_len=data_cfg.get("cont_len", 64),
            dataset_name=data_cfg["dataset_name"],
            dataset_config=data_cfg.get("dataset_config"),
            st5_encoder=st5,
            max_samples=data_cfg.get("max_samples"),
        )

    dataloader = build_dataloader(dataset, batch_size=g_cfg.get("batch_size", 256))

    # Model
    mlp = GuidanceMLP(
        d_plan=g_cfg.get("d_plan", 768),
        d_hidden=g_cfg.get("d_hidden", 1536),
        n_blocks=g_cfg.get("n_blocks", 4),
        attribute_dims=g_cfg.get("attribute_dims"),
    ).to(device)

    schedule = DiffusionSchedule()

    optimizer = AdamW(mlp.parameters(), lr=g_cfg.get("learning_rate", 1e-4))
    max_steps = g_cfg.get("max_steps", 50000)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    checkpoint_dir = g_cfg.get("checkpoint_dir", "checkpoints/guidance_mlp")

    tracking_on = init_tracking(cfg, job_name="train_guidance_mlp", tags=["guidance_mlp"])
    log.info(f"Wandb tracking: {'on' if tracking_on else 'off'}")

    log.info(f"Training guidance MLP for {max_steps} steps")
    mlp.train()

    data_iter = iter(dataloader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        z0 = batch["z0"].to(device)
        labels = {k: v.to(device) for k, v in batch["structure_labels"].items()}

        # Sample random timesteps and add noise
        t = torch.rand(z0.shape[0], device=device)
        z_t, _ = schedule.q_sample(z0, t)

        loss = mlp.compute_loss(z_t, t, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            loss_val = loss.item()
            log.info(f"step={step} loss={loss_val:.4f}")
            log_metrics({"train/loss": loss_val, "train/lr": scheduler.get_last_lr()[0]}, step=step)

        if step % 5000 == 0:
            save_dir = Path(checkpoint_dir) / f"step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(mlp.state_dict(), save_dir / "model.pt")
            with open(save_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2)
            log.info(f"Saved checkpoint → {save_dir}")

    finish_tracking()
    log.info("Guidance MLP training complete.")


if __name__ == "__main__":
    main()
