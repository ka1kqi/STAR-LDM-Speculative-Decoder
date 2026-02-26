#!/usr/bin/env python3
"""Train the hybrid diffusion + AR model.

Usage:
    python scripts/train_hybrid.py --config configs/default.yaml
    python scripts/train_hybrid.py --config configs/toy.yaml   # smoke test
    accelerate launch scripts/train_hybrid.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from src.models.hybrid import HybridModel
from src.data.dataset import FineWebDataset, ToyDataset, build_dataloader
from src.data.embedding import SentenceT5Encoder
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train_hybrid")


def save_checkpoint(model, optimizer, scheduler, step, cfg, save_dir):
    save_dir = Path(save_dir) / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(save_dir / "step.txt", "w") as f:
        f.write(str(step))
    log.info(f"Saved checkpoint at step {step} → {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    set_seed(train_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
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
        cfg_drop_prob=model_cfg.get("cfg_drop_prob", 0.1),
        final_noise_sigma2=model_cfg.get("final_noise_sigma2", 0.1),
    ).to(device)

    # Dataset
    if data_cfg.get("dataset_name") == "toy":
        dataset = ToyDataset(
            tokenizer,
            prefix_len=data_cfg.get("prefix_len", 32),
            cont_len=data_cfg.get("cont_len", 64),
            num_samples=data_cfg.get("max_samples", 256),
        )
        st5_encoder = None
    else:
        # ST5 runs on CPU intentionally — it's a frozen encoder used only in data
        # loading, not backpropagated through. Keeps GPU memory free for training.
        st5_encoder = SentenceT5Encoder(
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
            st5_encoder=st5_encoder,
            max_samples=data_cfg.get("max_samples"),
        )

    dataloader = build_dataloader(dataset, batch_size=train_cfg.get("batch_size", 16))

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.get("max_steps", 100000),
    )

    # Resume
    global_step = 0
    if args.resume:
        ckpt_dir = Path(args.resume)
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location=device))
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
        scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location=device))
        global_step = int((ckpt_dir / "step.txt").read_text().strip())
        log.info(f"Resumed from step {global_step}")

    # Mixed precision — detect dtype from model
    # bf16: use autocast but NO GradScaler (bf16 has enough dynamic range)
    # fp16: use autocast WITH GradScaler
    use_amp = train_cfg.get("fp16", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if model._dtype == torch.bfloat16 else torch.float16
    use_grad_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None
    if use_amp:
        log.info(f"AMP enabled with dtype={amp_dtype}, GradScaler={'on' if use_grad_scaler else 'off'}")

    max_steps = train_cfg.get("max_steps", 100000)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    save_every = train_cfg.get("save_every", 5000)
    log_every = train_cfg.get("log_every", 100)
    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints/hybrid")

    log.info(f"Training for {max_steps} steps, batch_size={train_cfg.get('batch_size')}, grad_accum={grad_accum}")

    model.train()
    optimizer.zero_grad()
    accum_loss = 0.0

    data_iter = iter(dataloader)
    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prefix_ids = batch["prefix_ids"].to(device)
        cont_ids = batch["cont_ids"].to(device)
        z0 = batch["z0"].to(device)
        structure_labels = {k: v.to(device) for k, v in batch["structure_labels"].items()}

        if use_amp:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(prefix_ids, cont_ids, z0, structure_labels)
                loss = outputs["loss"] / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            outputs = model(prefix_ids, cont_ids, z0, structure_labels)
            loss = outputs["loss"] / grad_accum
            loss.backward()

        accum_loss += loss.item()

        if (global_step + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % log_every == 0:
            log.info(
                f"step={global_step} loss={accum_loss:.4f} "
                f"loss_lm={outputs['loss_lm'].item():.4f} "
                f"loss_dm={outputs['loss_dm'].item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            accum_loss = 0.0

        if global_step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, cfg, checkpoint_dir)

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step, cfg, checkpoint_dir)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
