"""Streaming dataset for FineWeb and other HF text sources.

Handles chunking, prefix/continuation splitting, structure extraction,
and Sentence-T5 plan embedding computation.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Optional, Iterator

from .structure import extract_structure, canonical_serialize
from .embedding import SentenceT5Encoder


class FineWebDataset(IterableDataset):
    """Streaming dataset that yields (prefix_ids, cont_ids, z0_structure).

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
    prefix_len : int
        Number of tokens for the prefix.
    cont_len : int
        Number of tokens for the continuation.
    dataset_name : str
        HuggingFace dataset identifier (streaming mode).
    dataset_config : str | None
        Configuration name for the dataset.
    split : str
        Dataset split.
    text_column : str
        Name of the text column.
    st5_encoder : SentenceT5Encoder | None
        If provided, compute z0 embeddings on-the-fly; else return None.
    max_samples : int | None
        Cap the number of samples (for toy / debug runs).
    """

    def __init__(
        self,
        tokenizer,
        prefix_len: int = 32,
        cont_len: int = 64,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_config: Optional[str] = "sample-10BT",
        split: str = "train",
        text_column: str = "text",
        st5_encoder: Optional[SentenceT5Encoder] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.cont_len = cont_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_column = text_column
        self.st5_encoder = st5_encoder
        self.max_samples = max_samples
        self.total_len = prefix_len + cont_len

    def _stream(self) -> Iterator:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )
        count = 0
        for row in ds:
            text = row.get(self.text_column, "")
            if not text or len(text) < 50:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < self.total_len:
                continue

            prefix_ids = ids[: self.prefix_len]
            cont_ids = ids[self.prefix_len : self.total_len]

            cont_text = self.tokenizer.decode(cont_ids, skip_special_tokens=True)
            feats = extract_structure(cont_text)
            serial = canonical_serialize(feats)

            if self.st5_encoder is not None:
                z0 = self.st5_encoder.encode_single(serial)
            else:
                z0 = torch.zeros(768)  # CPU is fine here; .to(device) in training loop

            yield {
                "prefix_ids": torch.tensor(prefix_ids, dtype=torch.long),
                "cont_ids": torch.tensor(cont_ids, dtype=torch.long),
                "z0": z0,
                "structure_labels": feats.to_label_dict(),
            }

            count += 1
            if self.max_samples and count >= self.max_samples:
                return

    def __iter__(self):
        return self._stream()


class ToyDataset(IterableDataset):
    """Deterministic toy dataset for smoke-testing (no network needed)."""

    def __init__(self, tokenizer, prefix_len: int = 32, cont_len: int = 64,
                 num_samples: int = 256) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.cont_len = cont_len
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            vocab_size = self.tokenizer.vocab_size
            prefix_ids = torch.randint(100, min(vocab_size, 5000), (self.prefix_len,))
            cont_ids = torch.randint(100, min(vocab_size, 5000), (self.cont_len,))
            yield {
                "prefix_ids": prefix_ids,
                "cont_ids": cont_ids,
                "z0": torch.randn(768),
                "structure_labels": {"length_bucket": 1, "has_list": 0, "has_question": 0},
            }


def _collate(batch: list[dict]) -> dict:
    return {
        "prefix_ids": torch.stack([b["prefix_ids"] for b in batch]),
        "cont_ids": torch.stack([b["cont_ids"] for b in batch]),
        "z0": torch.stack([b["z0"] for b in batch]),
        "structure_labels": {
            k: torch.tensor([b["structure_labels"][k] for b in batch], dtype=torch.long)
            for k in batch[0]["structure_labels"]
        },
    }


def build_dataloader(
    dataset: IterableDataset,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
