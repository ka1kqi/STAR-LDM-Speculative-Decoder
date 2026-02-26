"""Lossless speculative decoding (Leviathan et al., 2022).

The draft model proposes K tokens; the target model verifies them in a single
forward pass.  Rejection sampling ensures the output distribution matches the
target model exactly (distribution-preserving).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


@torch.no_grad()
def lossless_speculative_decode(
    target_model,
    draft_model,
    input_ids: torch.Tensor,            # (1, L)
    max_new_tokens: int = 64,
    draft_k: int = 4,                   # tokens drafted per block
    top_p: float = 0.95,
    temperature: float = 1.0,
    repetition_penalty: float = 1.2,
    past_key_values_target=None,
    past_key_values_draft=None,
    soft_prompt_embeds: Optional[torch.Tensor] = None,  # (1, K, d_lm) plan conditioning
) -> dict:
    """Run lossless speculative decoding.

    Returns
    -------
    dict with keys:
        generated_ids : torch.Tensor — full sequence including input
        n_accepted : int — total accepted draft tokens
        n_drafted : int — total drafted tokens
        n_target_calls : int — number of target model forward calls
    """
    device = input_ids.device
    generated = input_ids.clone()
    n_accepted = 0
    n_drafted = 0
    n_target_calls = 0
    tokens_generated = 0

    # If soft prompt is provided, prepend it to the first forward call
    _first_call_target = True
    _first_call_draft = True

    def _get_probs(logits: torch.Tensor) -> torch.Tensor:
        logits = logits / max(temperature, 1e-8)
        return F.softmax(logits, dim=-1)

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

        # --- Draft phase: propose k tokens ---
        draft_tokens = []
        draft_probs_list = []
        draft_input = generated
        for _ in range(k):
            out_d = draft_model(draft_input[:, -1:] if not _first_call_draft else draft_input,
                                past_key_values=past_key_values_draft,
                                use_cache=True)
            _first_call_draft = False
            past_key_values_draft = out_d.past_key_values
            d_logits = out_d.logits[:, -1, :]
            d_probs = _get_probs(d_logits)
            tok_id = torch.multinomial(d_probs, 1)
            draft_tokens.append(tok_id)
            draft_probs_list.append(d_probs)
            draft_input = torch.cat([draft_input, tok_id], dim=-1)
        n_drafted += k

        draft_seq = torch.cat(draft_tokens, dim=-1)  # (1, k)
        candidate = torch.cat([generated, draft_seq], dim=-1)

        # --- Verify phase: single target model pass over all k+1 positions ---
        if _first_call_target and soft_prompt_embeds is not None:
            prefix_embeds = target_model.get_input_embeddings()(candidate)
            lm_input = torch.cat([soft_prompt_embeds, prefix_embeds], dim=1)
            out_t = target_model(inputs_embeds=lm_input, use_cache=True)
            _first_call_target = False
            past_key_values_target = None
            sp_len = soft_prompt_embeds.shape[1]
            t_logits = out_t.logits[:, sp_len + generated.shape[1] - 1:, :]
        else:
            verify_input = candidate[:, generated.shape[1] - 1:] if not _first_call_target else candidate
            out_t = target_model(verify_input if _first_call_target else candidate[:, -(k + 1):],
                                 past_key_values=past_key_values_target,
                                 use_cache=True)
            _first_call_target = False
            past_key_values_target = out_t.past_key_values
            t_logits = out_t.logits
        n_target_calls += 1

        # --- Rejection sampling ---
        # Batch-gather all target & draft probs for the k drafted tokens,
        # then transfer to CPU in one shot to avoid k separate GPU syncs.
        draft_tok_ids = draft_seq[0]  # (k,)
        t_probs_all = []
        d_probs_all = []
        for j in range(k):
            t_probs_all.append(_get_probs(t_logits[:, j, :]))
            d_probs_all.append(draft_probs_list[j])

        # Gather p(tok) for each drafted token — one GPU op per distribution
        p_target_k = torch.stack(
            [t_probs_all[j].squeeze(0)[draft_tok_ids[j]] for j in range(k)]
        )  # (k,)
        p_draft_k = torch.stack(
            [d_probs_all[j].squeeze(0)[draft_tok_ids[j]] for j in range(k)]
        )  # (k,)

        # Single CPU transfer: probabilities + uniform random draws
        rand_k = torch.rand(k, device=device)
        scalars = torch.stack([p_target_k, p_draft_k, rand_k]).cpu()  # (3, k)
        p_t_cpu = scalars[0]
        p_d_cpu = scalars[1]
        rand_cpu = scalars[2]

        accepted = 0
        for j in range(k):
            ratio = p_t_cpu[j].item() / max(p_d_cpu[j].item(), 1e-10)
            if rand_cpu[j].item() < min(1.0, ratio):
                accepted += 1
            else:
                # Reject: sample from adjusted distribution on GPU
                adjusted = (t_probs_all[j] - d_probs_all[j]).clamp(min=0)
                adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                correction_tok = torch.multinomial(adjusted, 1)
                generated = torch.cat([generated, draft_seq[:, :j], correction_tok], dim=-1)
                tokens_generated += j + 1
                n_accepted += j
                past_key_values_draft = None
                _first_call_draft = True
                past_key_values_target = None
                _first_call_target = True
                break
        else:
            # All k tokens accepted; sample bonus token from target at position k
            bonus_probs = _get_probs(t_logits[:, k, :])
            bonus_tok = torch.multinomial(bonus_probs, 1)
            generated = torch.cat([generated, draft_seq, bonus_tok], dim=-1)
            tokens_generated += k + 1
            n_accepted += k

    return {
        "generated_ids": generated,
        "n_accepted": n_accepted,
        "n_drafted": n_drafted,
        "n_target_calls": n_target_calls,
    }
