"""Sentence-T5 encoder with optional disk caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class SentenceT5Encoder:
    """Wrapper around sentence-transformers/sentence-t5-xl (768-dim).

    Provides batch encoding and a simple file-based cache.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/sentence-t5-xl",
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = 768  # sentence-t5-xl output dim
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _try_load(self, text: str) -> Optional[np.ndarray]:
        key = self._cache_key(text)
        if key in self._mem_cache:
            return self._mem_cache[key]
        if self.cache_dir:
            path = self.cache_dir / f"{key}.npy"
            if path.exists():
                arr = np.load(path)
                self._mem_cache[key] = arr
                return arr
        return None

    def _store(self, text: str, emb: np.ndarray) -> None:
        key = self._cache_key(text)
        self._mem_cache[key] = emb
        if self.cache_dir:
            np.save(self.cache_dir / f"{key}.npy", emb)

    # ------------------------------------------------------------------
    def encode(self, texts: list[str], batch_size: int = 64) -> torch.Tensor:
        """Return (N, 768) float32 tensor of embeddings."""
        results: list[np.ndarray] = []
        to_encode: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cached = self._try_load(t)
            if cached is not None:
                results.append(cached)
            else:
                to_encode.append((i, t))
                results.append(np.empty(0))  # placeholder

        if to_encode:
            batch_texts = [t for _, t in to_encode]
            embs = self.model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
            for (idx, txt), emb in zip(to_encode, embs):
                self._store(txt, emb)
                results[idx] = emb

        return torch.tensor(np.stack(results), dtype=torch.float32)

    def encode_single(self, text: str) -> torch.Tensor:
        return self.encode([text])[0]
