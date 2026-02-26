"""StoryCloze LLM-as-judge evaluation harness.

Uses a pluggable LLM backend (default: Claude 3.7 Sonnet via Anthropic API)
to judge which continuation is better: the model's generation vs. a reference.

The prompt template is loaded from an editable file.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from ..utils.logging import get_logger

log = get_logger(__name__)

# Default prompt template path (editable)
DEFAULT_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "configs" / "storycloze_judge_template.txt"


def load_judge_template(path: Optional[str] = None) -> str:
    """Load the judge prompt template from file."""
    p = Path(path) if path else DEFAULT_TEMPLATE_PATH
    if p.exists():
        return p.read_text()
    # Fallback inline template
    return (
        "You are an expert story evaluator. Given a story context and two possible "
        "continuations (A and B), determine which continuation is more coherent, "
        "engaging, and natural.\n\n"
        "Context:\n{context}\n\n"
        "Continuation A:\n{cont_a}\n\n"
        "Continuation B:\n{cont_b}\n\n"
        "Which continuation is better? Answer with exactly one of: A or B.\n"
        "Your answer:"
    )


def _call_anthropic(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic API. Requires ANTHROPIC_API_KEY env var."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API. Requires OPENAI_API_KEY env var."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=8,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


BACKENDS = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}


def _bootstrap_ci(wins: list[int], n_bootstrap: int = 1000,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for win rate."""
    arr = np.array(wins, dtype=float)
    mean = arr.mean()
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(sample.mean())
    boot_means = sorted(boot_means)
    lo = boot_means[int(alpha / 2 * n_bootstrap)]
    hi = boot_means[int((1 - alpha / 2) * n_bootstrap)]
    return float(mean), float(lo), float(hi)


def evaluate_storycloze(
    pipeline,
    tokenizer,
    n_samples: int = 200,
    prefix_len: int = 32,
    cont_len: int = 64,
    backend: str = "anthropic",
    judge_model: Optional[str] = None,
    template_path: Optional[str] = None,
    device: str = "cuda",
) -> dict:
    """Run StoryCloze-style LLM-as-judge evaluation.

    Uses the Story Cloze Test validation split (ROCStories).

    Returns dict with: win_rate, ci_lo, ci_hi, n_samples, raw_judgments.
    """
    from datasets import load_dataset

    log.info("Loading StoryCloze / ROCStories data...")
    try:
        ds = load_dataset("story_cloze", "2016", split="validation", trust_remote_code=True)
    except Exception:
        log.warning("StoryCloze dataset not available; using synthetic contexts.")
        ds = None

    template = load_judge_template(template_path)
    call_fn = BACKENDS.get(backend, _call_anthropic)
    judge_kwargs = {}
    if judge_model:
        judge_kwargs["model"] = judge_model

    wins: list[int] = []
    judgments: list[dict] = []

    for i in tqdm(range(n_samples), desc="StoryCloze eval"):
        if ds is not None and i < len(ds):
            row = ds[i]
            context = row.get("input_sentence_1", "") + " " + \
                      row.get("input_sentence_2", "") + " " + \
                      row.get("input_sentence_3", "") + " " + \
                      row.get("input_sentence_4", "")
            reference = row.get("sentence_quiz1", row.get("sentence_quiz2", ""))
        else:
            context = f"Once upon a time, in a land far away, there lived a brave adventurer (story {i})."
            reference = "The adventurer set out on a journey to find the lost treasure."

        ctx_ids = tokenizer.encode(context, add_special_tokens=False)[:prefix_len]
        if len(ctx_ids) < prefix_len:
            ctx_ids = ctx_ids + [tokenizer.pad_token_id or 0] * (prefix_len - len(ctx_ids))
        prefix_ids = torch.tensor([ctx_ids], device=device)

        result = pipeline(prefix_ids)
        gen_ids = result["generated_ids"][0, prefix_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Randomize position to avoid bias
        if random.random() < 0.5:
            cont_a, cont_b = gen_text, reference
            model_is_a = True
        else:
            cont_a, cont_b = reference, gen_text
            model_is_a = False

        prompt = template.format(context=context, cont_a=cont_a, cont_b=cont_b)

        try:
            answer = call_fn(prompt, **judge_kwargs)
            model_wins = (answer.upper().startswith("A") and model_is_a) or \
                         (answer.upper().startswith("B") and not model_is_a)
            wins.append(1 if model_wins else 0)
            judgments.append({
                "context": context[:200],
                "gen": gen_text[:200],
                "ref": reference[:200],
                "judge_answer": answer,
                "model_wins": model_wins,
            })
        except Exception as e:
            log.warning(f"Judge call failed for sample {i}: {e}")
            continue

    if not wins:
        return {"win_rate": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_samples": 0, "raw_judgments": []}

    mean, lo, hi = _bootstrap_ci(wins)
    log.info(f"Win rate: {mean:.3f} [{lo:.3f}, {hi:.3f}] (n={len(wins)})")

    return {
        "win_rate": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_samples": len(wins),
        "raw_judgments": judgments,
    }
