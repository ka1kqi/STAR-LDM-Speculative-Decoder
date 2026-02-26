"""Plotting utilities for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_eval_plots(
    results_path: str,
    output_dir: str = "plots",
    prefix: str = "eval",
) -> list[str]:
    """Generate evaluation plots from a JSON results file.

    Returns list of saved plot file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    saved: list[str] = []

    # 1) Bar chart of main metrics
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = {}
    for key in ["mauve", "gen_ppl", "lexical_diversity"]:
        if key in results and results[key] is not None:
            metrics[key] = results[key]

    if metrics:
        names = list(metrics.keys())
        values = list(metrics.values())
        ax.bar(names, values, color=["#2196F3", "#FF9800", "#4CAF50"][:len(names)])
        ax.set_title("C4 Evaluation Metrics")
        ax.set_ylabel("Value")
        for i, v in enumerate(values):
            ax.text(i, v + 0.01 * max(values), f"{v:.3f}", ha="center")
        path = str(out / f"{prefix}_c4_metrics.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)

    # 2) StoryCloze win-rate with CI
    if "win_rate" in results:
        fig, ax = plt.subplots(figsize=(5, 4))
        wr = results["win_rate"]
        lo = results.get("ci_lo", wr)
        hi = results.get("ci_hi", wr)
        ax.bar(["Model"], [wr], yerr=[[wr - lo], [hi - wr]], capsize=8, color="#9C27B0")
        ax.axhline(0.5, color="gray", linestyle="--", label="Random")
        ax.set_ylabel("Win Rate")
        ax.set_title("StoryCloze LLM-as-Judge Win Rate")
        ax.set_ylim(0, 1)
        ax.legend()
        path = str(out / f"{prefix}_storycloze_winrate.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        plt.close(fig)

    # 3) Spec decode stats
    if "spec_stats" in results and results["spec_stats"]:
        stats = results["spec_stats"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        if "acceptance_rate" in stats:
            axes[0].bar(["Acceptance Rate"], [stats["acceptance_rate"]], color="#00BCD4")
            axes[0].set_ylim(0, 1)
            axes[0].set_title("Spec Decode Acceptance Rate")

        if "tokens_per_sec" in stats:
            axes[1].bar(["Tokens/sec"], [stats["tokens_per_sec"]], color="#FF5722")
            axes[1].set_title("Generation Throughput")

        path = str(out / f"{prefix}_spec_decode.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        plt.close(fig)

    # 4) Diffusion steps sweep (if present)
    if "steps_sweep" in results:
        sweep = results["steps_sweep"]
        fig, ax = plt.subplots(figsize=(7, 4))
        steps = [s["steps"] for s in sweep]
        mauve_vals = [s.get("mauve", 0) for s in sweep]
        ax.plot(steps, mauve_vals, "o-", color="#E91E63")
        ax.set_xlabel("Diffusion Steps")
        ax.set_ylabel("MAUVE")
        ax.set_title("MAUVE vs. Diffusion Steps")
        path = str(out / f"{prefix}_steps_sweep.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(path)
        plt.close(fig)

    return saved
