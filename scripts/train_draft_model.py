#!/usr/bin/env python3
"""Train / finetune a draft AR model for speculative decoding.

The draft model shares the same tokenizer and architecture as the target.
It is trained (or distilled) to approximate the target model's distribution.

Usage:
    python scripts/train_draft_model.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import FineWebDataset, ToyDataset, build_dataloader
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train_draft")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    draft_cfg = cfg.get("draft", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    set_seed(cfg.get("training", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Draft model (same architecture, initialized from pretrained)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_cfg.get("model_name", model_cfg["ar_model_name"]),
        trust_remote_code=True,
    ).to(device)

    # Dataset
    if data_cfg.get("dataset_name") == "toy":
        dataset = ToyDataset(tokenizer, prefix_len=data_cfg.get("prefix_len", 32),
                             cont_len=data_cfg.get("cont_len", 64))
    else:
        dataset = FineWebDataset(
            tokenizer=tokenizer,
            prefix_len=data_cfg.get("prefix_len", 32),
            cont_len=data_cfg.get("cont_len", 64),
            dataset_name=data_cfg["dataset_name"],
            dataset_config=data_cfg.get("dataset_config"),
            st5_encoder=None,
            max_samples=data_cfg.get("max_samples"),
        )

    dataloader = build_dataloader(dataset, batch_size=draft_cfg.get("batch_size", 32))

    optimizer = AdamW(draft_model.parameters(), lr=draft_cfg.get("learning_rate", 5e-5))
    max_steps = draft_cfg.get("max_steps", 50000)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    checkpoint_dir = draft_cfg.get("checkpoint_dir", "checkpoints/draft")

    log.info(f"Training draft model for {max_steps} steps")
    draft_model.train()

    data_iter = iter(dataloader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Train on full prefix + continuation
        input_ids = torch.cat([batch["prefix_ids"], batch["cont_ids"]], dim=1).to(device)
        labels = input_ids.clone()
        labels[:, :batch["prefix_ids"].shape[1]] = -100  # mask prefix

        outputs = draft_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            log.info(f"step={step} loss={loss.item():.4f}")

        if step % 5000 == 0:
            save_dir = Path(checkpoint_dir) / f"step_{step}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(draft_model.state_dict(), save_dir / "model.pt")
            with open(save_dir / "config.json", "w") as f:
                json.dump(cfg, f, indent=2)
            log.info(f"Saved checkpoint → {save_dir}")

    log.info("Draft model training complete.")


if __name__ == "__main__":
    main()
