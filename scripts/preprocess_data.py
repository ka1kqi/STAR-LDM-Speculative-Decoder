#!/usr/bin/env python3
"""Preprocess data: stream FineWeb, extract structures, cache Sentence-T5 embeddings.

Usage:
    python scripts/preprocess_data.py --config configs/default.yaml --max-samples 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.data.structure import extract_structure, canonical_serialize
from src.data.embedding import SentenceT5Encoder
from src.utils.config import load_config, merge_configs
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("preprocess")


def main():
    parser = argparse.ArgumentParser(description="Preprocess FineWeb data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="data/preprocessed")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))

    data_cfg = cfg.get("data", {})
    prefix_len = data_cfg.get("prefix_len", 32)
    cont_len = data_cfg.get("cont_len", 64)
    total_len = prefix_len + cont_len

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["ar_model_name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    st5 = SentenceT5Encoder(
        model_name=data_cfg.get("st5_model", "sentence-transformers/sentence-t5-xl"),
        cache_dir=data_cfg.get("st5_cache_dir", "cache/st5_embeddings"),
        device=args.device,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = args.max_samples or data_cfg.get("max_samples")
    dataset_name = data_cfg.get("dataset_name", "HuggingFaceFW/fineweb")
    dataset_config = data_cfg.get("dataset_config", "sample-10BT")

    log.info(f"Streaming {dataset_name} ({dataset_config}), max_samples={max_samples}")

    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split="train",
                      streaming=True, trust_remote_code=True)

    metadata: list[dict] = []
    embeddings: list[torch.Tensor] = []
    batch_texts: list[str] = []
    count = 0

    for row in ds:
        text = row.get(data_cfg.get("text_column", "text"), "")
        if not text or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < total_len:
            continue

        cont_ids = ids[prefix_len:total_len]
        cont_text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        feats = extract_structure(cont_text)
        serial = canonical_serialize(feats)

        metadata.append({
            "prefix_ids": ids[:prefix_len],
            "cont_ids": cont_ids,
            "structure": feats.to_label_dict(),
            "serial": serial,
        })
        batch_texts.append(serial)

        count += 1
        if count % 1000 == 0:
            log.info(f"Processed {count} samples...")

        if max_samples and count >= max_samples:
            break

    log.info(f"Encoding {len(batch_texts)} structure serializations with Sentence-T5...")
    z0_all = st5.encode(batch_texts, batch_size=64)

    log.info(f"Saving to {out_dir}...")
    torch.save(z0_all, out_dir / "z0_embeddings.pt")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    log.info(f"Done. {count} samples preprocessed.")


if __name__ == "__main__":
    main()
