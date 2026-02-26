"""C4 validation evaluation: MAUVE, generative perplexity, lexical diversity."""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm

from ..utils.logging import get_logger

log = get_logger(__name__)


def _compute_lexical_diversity(texts: list[str], max_n: int = 4) -> float:
    """Product of unique n-gram ratios for n=2..max_n."""
    diversity = 1.0
    for n in range(2, max_n + 1):
        all_ngrams: list[tuple] = []
        for text in texts:
            tokens = text.split()
            all_ngrams.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        if len(all_ngrams) > 0:
            ratio = len(set(all_ngrams)) / len(all_ngrams)
            diversity *= ratio
    return diversity


def _compute_generative_perplexity(
    texts: list[str],
    model_name: str = "meta-llama/Llama-3.2-3B",
    batch_size: int = 8,
    max_length: int = 256,
    device: str = "cuda",
) -> float:
    """Compute generative perplexity using a reference LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    log.info(f"Loading perplexity model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**enc)
            shift_logits = out.logits[:, :-1, :]
            shift_labels = enc["input_ids"][:, 1:]
            mask = (shift_labels != tokenizer.pad_token_id).float()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(shift_labels.shape)
            total_loss += (loss * mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return float(np.exp(avg_loss))


def _compute_mauve(reference_texts: list[str], generated_texts: list[str],
                   device_id: int = 0) -> float:
    """Compute MAUVE score between reference and generated texts."""
    import mauve

    result = mauve.compute_mauve(
        p_text=reference_texts,
        q_text=generated_texts,
        device_id=device_id,
        max_text_length=256,
        verbose=False,
    )
    return result.mauve


def evaluate_c4(
    pipeline,
    tokenizer,
    n_samples: int = 5000,
    prefix_len: int = 32,
    cont_len: int = 64,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    split: str = "validation",
    ppl_model: str = "meta-llama/Llama-3.2-3B",
    device: str = "cuda",
    compute_ppl: bool = True,
    batch_size: int = 16,
) -> dict:
    """Run C4 evaluation.

    Returns dict with: mauve, gen_ppl, lexical_diversity, n_samples.
    """
    from datasets import load_dataset

    log.info(f"Loading C4 {split} split...")
    ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True,
                      trust_remote_code=True)

    reference_texts: list[str] = []
    generated_texts: list[str] = []
    count = 0

    for row in tqdm(ds, total=n_samples, desc="C4 eval"):
        text = row.get("text", "")
        if not text or len(text) < 50:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < prefix_len + cont_len:
            continue

        prefix_ids = torch.tensor([ids[:prefix_len]], device=device)
        ref_cont = tokenizer.decode(ids[prefix_len:prefix_len + cont_len], skip_special_tokens=True)

        result = pipeline(prefix_ids)
        gen_ids = result["generated_ids"][0, prefix_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        reference_texts.append(ref_cont)
        generated_texts.append(gen_text)

        count += 1
        if count >= n_samples:
            break

    log.info(f"Collected {count} samples. Computing metrics...")

    metrics: dict = {"n_samples": count}

    # MAUVE
    try:
        metrics["mauve"] = _compute_mauve(reference_texts, generated_texts)
        log.info(f"MAUVE: {metrics['mauve']:.4f}")
    except Exception as e:
        log.warning(f"MAUVE computation failed: {e}")
        metrics["mauve"] = None

    # Generative perplexity
    if compute_ppl:
        try:
            metrics["gen_ppl"] = _compute_generative_perplexity(
                generated_texts, model_name=ppl_model, device=device
            )
            log.info(f"Gen PPL: {metrics['gen_ppl']:.2f}")
        except Exception as e:
            log.warning(f"PPL computation failed: {e}")
            metrics["gen_ppl"] = None

    # Lexical diversity
    metrics["lexical_diversity"] = _compute_lexical_diversity(generated_texts)
    log.info(f"Lexical diversity: {metrics['lexical_diversity']:.4f}")

    return metrics
