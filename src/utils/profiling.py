"""Simple profiling helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import torch


class Timer:
    """Accumulating wall-clock timer."""

    def __init__(self) -> None:
        self._total = 0.0
        self._count = 0
        self._start: float | None = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        assert self._start is not None
        elapsed = time.perf_counter() - self._start
        self._total += elapsed
        self._count += 1
        self._start = None
        return elapsed

    @property
    def total(self) -> float:
        return self._total

    @property
    def avg(self) -> float:
        return self._total / max(self._count, 1)

    @property
    def count(self) -> int:
        return self._count


@contextmanager
def profile_cuda(label: str = "op") -> Generator[dict, None, None]:
    """Context manager that records CUDA elapsed time (ms) into *stats*."""
    stats: dict = {"label": label}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield stats
        end.record()
        torch.cuda.synchronize()
        stats["cuda_ms"] = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        yield stats
        stats["cuda_ms"] = (time.perf_counter() - t0) * 1000.0
