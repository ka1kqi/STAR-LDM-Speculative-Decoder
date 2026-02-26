"""Approximate speculative decoding with acceptance threshold τ(c) tied to plan logSNR.

When the diffusion plan confidence is high (large logSNR → small residual noise),
we trust the draft model more and raise the acceptance threshold, allowing more
aggressive speculation.  When confidence is low, we lower the threshold to reject
more aggressively and fall back to the target model.

τ(c) = τ_min + (τ_max - τ_min) · σ(c · logSNR)

where c is a learned or fixed scaling factor.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Optional


def _compute_tau(logsnr: float, tau_min: float = 0.3, tau_max: float = 0.95,
                 scale: float = 1.0) -> float:
    """Compute acceptance threshold from plan logSNR (pure Python, no GPU)."""
    sigmoid_val = 1.0 / (1.0 + math.exp(-scale * logsnr))
    return tau_min + (tau_max - tau_min) * sigmoid_val


@torch.no_grad()
def approximate_speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,
    plan_logsnr: float,
    max_new_tokens: int = 64,
    draft_k: int = 4,
    top_p: float = 0.95,
    temperature: float = 1.0,
    tau_min: float = 0.3,
    tau_max: float = 0.95,
    tau_scale: float = 1.0,
    soft_prompt_embeds: Optional[torch.Tensor] = None,
) -> dict:
    """Approximate speculative decoding with logSNR-adaptive threshold.

    Instead of exact rejection sampling, we accept a draft token if
    p_target(tok) >= τ(c), where τ is tied to the plan confidence.

    Returns
    -------
    dict with same keys as lossless version, plus:
        tau : float — the computed threshold
    """
    device = input_ids.device
    tau = _compute_tau(plan_logsnr, tau_min, tau_max, tau_scale)

    generated = input_ids.clone()
    n_accepted = 0
    n_drafted = 0
    n_target_calls = 0
    tokens_generated = 0

    past_kv_draft = None
    past_kv_target = None

    def _get_probs(logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / max(temperature, 1e-8), dim=-1)

    def _nucleus_sample(probs: torch.Tensor) -> torch.Tensor:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        tok_pos = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, tok_pos)

    while tokens_generated < max_new_tokens:
        k = min(draft_k, max_new_tokens - tokens_generated)
        if k <= 0:
            break

        # Draft phase
        draft_tokens = []
        cur = generated
        for _ in range(k):
            inp = cur if past_kv_draft is None else cur[:, -1:]
            out_d = draft_model(inp, past_key_values=past_kv_draft, use_cache=True)
            past_kv_draft = out_d.past_key_values
            d_probs = _get_probs(out_d.logits[:, -1, :])
            tok = _nucleus_sample(d_probs)
            draft_tokens.append(tok)
            cur = torch.cat([cur, tok], dim=-1)
        n_drafted += k

        draft_seq = torch.cat(draft_tokens, dim=-1)
        candidate = torch.cat([generated, draft_seq], dim=-1)

        # Verify phase
        verify_inp = candidate if past_kv_target is None else candidate[:, -(k + 1):]

        if past_kv_target is None and soft_prompt_embeds is not None:
            prefix_embeds = target_model.get_input_embeddings()(candidate)
            lm_input = torch.cat([soft_prompt_embeds, prefix_embeds], dim=1)
            out_t = target_model(inputs_embeds=lm_input, use_cache=True)
            sp_len = soft_prompt_embeds.shape[1]
            t_logits = out_t.logits[:, sp_len + generated.shape[1] - 1:, :]
            past_kv_target = out_t.past_key_values
        else:
            out_t = target_model(verify_inp, past_key_values=past_kv_target, use_cache=True)
            past_kv_target = out_t.past_key_values
            t_logits = out_t.logits
        n_target_calls += 1

        # Approximate acceptance: batch-gather p_target for all k tokens,
        # transfer to CPU once, then loop on CPU scalars.
        draft_tok_ids = draft_seq[0]  # (k,)
        t_probs_all = [_get_probs(t_logits[:, j, :]) for j in range(k)]
        p_target_k = torch.stack(
            [t_probs_all[j].squeeze(0)[draft_tok_ids[j]] for j in range(k)]
        )  # (k,)
        p_target_cpu = p_target_k.cpu()  # single GPU→CPU transfer

        accepted = 0
        for j in range(k):
            if p_target_cpu[j].item() >= tau:
                accepted += 1
            else:
                # Reject: resample from target on GPU
                correction = _nucleus_sample(t_probs_all[j])
                generated = torch.cat([generated, draft_seq[:, :j], correction], dim=-1)
                tokens_generated += j + 1
                n_accepted += j
                past_kv_draft = None
                past_kv_target = None
                break
        else:
            # All accepted + bonus token
            bonus_probs = _get_probs(t_logits[:, k, :])
            bonus = _nucleus_sample(bonus_probs)
            generated = torch.cat([generated, draft_seq, bonus], dim=-1)
            tokens_generated += k + 1
            n_accepted += k

    return {
        "generated_ids": generated,
        "n_accepted": n_accepted,
        "n_drafted": n_drafted,
        "n_target_calls": n_target_calls,
        "tau": tau,
    }
