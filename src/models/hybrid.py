"""HybridModel: the full Stop–Think–AutoRegress architecture.

Wraps the AR backbone (Qwen2.5-0.5B), PromptEncoder (DiT1),
DiffusionPrediction (DiT2), and soft-prompt projection.

Training produces:
    L_total = L_LM + β·L_DM + γ·L_align
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dit import PromptEncoder, DiffusionPrediction
from .diffusion import DiffusionSchedule, sigmoid_weight


class HybridModel(nn.Module):
    """STAR-LDM hybrid diffusion + AR model.

    Parameters
    ----------
    ar_model_name : str — HuggingFace model id for the AR backbone.
    d_plan : int — plan embedding dimension (768).
    d_dit : int — DiT hidden dimension (1024).
    dit_layers : int — number of layers per DiT module (6).
    dit_heads : int — attention heads (16).
    dit_head_dim : int — per-head dimension (64).
    soft_prompt_len : int — K, number of soft prompt tokens (8).
    beta : float — diffusion loss weight (5.0).
    gamma : float — alignment loss weight (default 0.0).
    cfg_drop_prob : float — probability of dropping prefix for CFG (0.1).
    final_noise_sigma2 : float — optional final noise injection variance (0.1).
    """

    def __init__(
        self,
        ar_model_name: str = "Qwen/Qwen2.5-0.5B",
        d_plan: int = 768,
        d_dit: int = 1024,
        dit_layers: int = 6,
        dit_heads: int = 16,
        dit_head_dim: int = 64,
        soft_prompt_len: int = 8,
        beta: float = 5.0,
        gamma: float = 0.0,
        cfg_drop_prob: float = 0.1,
        final_noise_sigma2: float = 0.1,
    ):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.cfg_drop_prob = cfg_drop_prob
        self.final_noise_sigma2 = final_noise_sigma2
        self.soft_prompt_len = soft_prompt_len
        self.d_plan = d_plan

        # AR backbone — freeze initially; finetune later if desired
        self.ar = AutoModelForCausalLM.from_pretrained(ar_model_name, trust_remote_code=True)
        self.d_lm = self.ar.config.hidden_size  # do NOT hardcode

        # Determine compute dtype — match AR backbone (typically bf16)
        self._dtype = next(self.ar.parameters()).dtype

        # Soft-prompt projector: DiT output → K soft prompt tokens in LM space
        self.soft_prompt_proj = nn.Sequential(
            nn.Linear(d_dit, d_dit),
            nn.SiLU(),
            nn.Linear(d_dit, self.d_lm * soft_prompt_len),
        ).to(self._dtype)

        # Null-context embedding (learnable) for CFG
        self.null_context = nn.Parameter(
            (torch.randn(1, 1, d_dit) * 0.02).to(self._dtype)
        )

        # DiT modules — bf16 to match AR backbone
        self.dit1 = PromptEncoder(
            d_lm=self.d_lm, d_plan=d_plan, d_dit=d_dit,
            n_layers=dit_layers, heads=dit_heads, head_dim=dit_head_dim,
        ).to(self._dtype)
        self.dit2 = DiffusionPrediction(
            d_plan=d_plan, d_dit=d_dit,
            n_layers=dit_layers, heads=dit_heads, head_dim=dit_head_dim,
        ).to(self._dtype)

        # Diffusion schedule (math stays in fp32 for numerical stability;
        # inputs/outputs are cast at call sites)
        self.schedule = DiffusionSchedule()

        # Optional alignment head: predict structure attributes from decoder hidden states
        self.align_head = (
            nn.Linear(self.d_lm, 8).to(self._dtype) if gamma > 0 else None
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_prefix_hidden(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Run AR model on prefix and return last-layer hidden states (bf16)."""
        out = self.ar(prefix_ids, output_hidden_states=True)
        return out.hidden_states[-1]  # (B, Lp, d_lm) — stays in AR dtype (bf16)

    def _make_soft_prompt(self, h1_pooled: torch.Tensor) -> torch.Tensor:
        """Project DiT1 output to soft prompt tokens.
        h1_pooled: (B, d_dit) — e.g. mean-pool of h1.
        Returns: (B, K, d_lm)
        """
        B = h1_pooled.shape[0]
        sp = self.soft_prompt_proj(h1_pooled)
        return sp.view(B, self.soft_prompt_len, self.d_lm)

    def _null_context_expanded(self, B: int, L: int, device: torch.device) -> torch.Tensor:
        """Expand null context to (B, L, d_dit)."""
        return self.null_context.expand(B, L, -1).to(device)

    # ------------------------------------------------------------------
    # Forward: training
    # ------------------------------------------------------------------

    def forward(
        self,
        prefix_ids: torch.Tensor,   # (B, Lp)
        cont_ids: torch.Tensor,     # (B, Lc)
        z0: torch.Tensor,           # (B, D_plan)
        structure_labels: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        B, Lp = prefix_ids.shape
        Lc = cont_ids.shape[1]
        device = prefix_ids.device

        # --- Prefix encoding ---
        prefix_hidden = self._get_prefix_hidden(prefix_ids)  # (B, Lp, d_lm)

        # --- Diffusion forward process ---
        # Schedule math runs in fp32; cast results to model dtype for DiT
        t = torch.rand(B, device=device)
        z0_f32 = z0.float()
        z_t_f32, noise_f32 = self.schedule.q_sample(z0_f32, t)
        z_t = z_t_f32.to(self._dtype)
        noise = noise_f32

        # --- DiT1 ---
        # CFG: drop prefix with probability cfg_drop_prob
        if self.training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob
            null_ctx = self._null_context_expanded(B, Lp, device)
            # Replace prefix_hidden with null for dropped samples
            prefix_hidden_dit = torch.where(
                drop_mask.view(B, 1, 1).expand_as(prefix_hidden),
                # Use a zero projection through proj_prefix equivalent
                torch.zeros_like(prefix_hidden),
                prefix_hidden,
            )
        else:
            prefix_hidden_dit = prefix_hidden

        h1 = self.dit1(prefix_hidden_dit, z_t, t)  # (B, Lp+1, d_dit)
        h1_pooled = h1.mean(dim=1)                  # (B, d_dit)

        # --- DiT2: v-prediction ---
        # For CFG training: compute null context for skip
        null_ctx_h1 = None
        if self.training and self.cfg_drop_prob > 0:
            null_prefix = torch.zeros_like(prefix_hidden)
            null_h1 = self.dit1(null_prefix, z_t, t)
            null_ctx_h1 = null_h1

        v_pred = self.dit2(z_t, h1, t, null_context=null_ctx_h1)

        # --- Diffusion loss (v-prediction MSE weighted by sigmoid(logSNR)) ---
        # Compute in fp32 for numerical stability
        v_target = self.schedule.v_target(z0_f32, noise_f32, t)
        logsnr = self.schedule.logsnr(t)
        weight = sigmoid_weight(logsnr).view(B, 1)
        loss_dm = (weight * (v_pred.float() - v_target).pow(2)).mean()

        # --- Soft prompt → LM continuation loss ---
        soft_prompt = self._make_soft_prompt(h1_pooled)  # (B, K, d_lm)

        # Build LM inputs: [soft_prompt_tokens | cont_tokens]
        # Both soft_prompt and cont_embeds are in self._dtype (bf16)
        cont_embeds = self.ar.get_input_embeddings()(cont_ids)  # (B, Lc, d_lm)
        lm_inputs = torch.cat([soft_prompt, cont_embeds[:, :-1, :]], dim=1)  # (B, K+Lc-1, d_lm)

        lm_out = self.ar(inputs_embeds=lm_inputs, output_hidden_states=self.gamma > 0)
        lm_logits = lm_out.logits  # (B, K+Lc-1, vocab)

        # LM loss: only on continuation positions
        cont_logits = lm_logits[:, self.soft_prompt_len - 1:, :]  # (B, Lc, vocab)
        loss_lm = F.cross_entropy(
            cont_logits.reshape(-1, cont_logits.shape[-1]),
            cont_ids.reshape(-1),
            ignore_index=-100,
        )

        # --- Prefix LM loss (standard autoregressive on prefix) ---
        prefix_lm_out = self.ar(prefix_ids)
        prefix_logits = prefix_lm_out.logits[:, :-1, :]
        prefix_targets = prefix_ids[:, 1:]
        loss_prefix_lm = F.cross_entropy(
            prefix_logits.reshape(-1, prefix_logits.shape[-1]),
            prefix_targets.reshape(-1),
            ignore_index=-100,
        )

        loss_lm_total = loss_prefix_lm + loss_lm

        # --- Alignment loss (optional) ---
        loss_align = torch.tensor(0.0, device=device)
        if self.gamma > 0 and self.align_head is not None and structure_labels is not None:
            hidden_states = lm_out.hidden_states[-1]
            align_logits = self.align_head(hidden_states.mean(dim=1))
            # Simple regression/classification placeholder
            # Here we just MSE toward a target; expand as needed
            for name, labels in structure_labels.items():
                if name == "length_bucket":
                    loss_align = loss_align + F.cross_entropy(
                        align_logits[:, :4], labels
                    )

        # --- Total loss ---
        loss_total = loss_lm_total + self.beta * loss_dm + self.gamma * loss_align

        return {
            "loss": loss_total,
            "loss_lm": loss_lm_total,
            "loss_dm": loss_dm,
            "loss_align": loss_align,
            "v_pred": v_pred.detach(),
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def denoise_fn(self, z_t: torch.Tensor, t: torch.Tensor,
                   prefix_hidden: torch.Tensor,
                   cfg_scale: float = 1.0) -> torch.Tensor:
        """Single denoising step: returns v_pred, optionally with CFG."""
        B = z_t.shape[0]
        z_t_bf = z_t.to(self._dtype)
        h1 = self.dit1(prefix_hidden, z_t_bf, t)

        if cfg_scale > 1.0:
            null_prefix = torch.zeros_like(prefix_hidden)
            null_h1 = self.dit1(null_prefix, z_t, t)
            v_cond = self.dit2(z_t_bf, h1, t)
            v_uncond = self.dit2(z_t_bf, null_h1, t, null_context=null_h1)
            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = self.dit2(z_t_bf, h1, t)

        return v_pred

    @torch.no_grad()
    def plan(self, prefix_ids: torch.Tensor, sampler, cfg_scale: float = 1.0,
             guidance_mlp=None, guidance_labels=None,
             guidance_scale: float = 0.0) -> torch.Tensor:
        """Run the diffusion planner to produce z_hat from prefix.

        Returns z_hat: (B, D_plan).
        """
        device = prefix_ids.device
        B = prefix_ids.shape[0]
        prefix_hidden = self._get_prefix_hidden(prefix_ids)

        def _denoise(z_t, t, **kw):
            v = self.denoise_fn(z_t, t, prefix_hidden, cfg_scale=cfg_scale)
            # Classifier guidance
            if guidance_mlp is not None and guidance_scale > 0.0 and guidance_labels is not None:
                grad = guidance_mlp.classifier_gradient(z_t, t, guidance_labels)
                v = v - guidance_scale * grad  # gradient ascent on log p(y|z_t)
            return v

        z_hat = sampler.sample(
            denoise_fn=_denoise,
            shape=(B, self.d_plan),
            device=device,
        )

        # Optional final noise injection
        if self.final_noise_sigma2 > 0:
            z_hat = z_hat + (self.final_noise_sigma2 ** 0.5) * torch.randn_like(z_hat)

        return z_hat

    @torch.no_grad()
    def generate_from_plan(
        self,
        prefix_ids: torch.Tensor,
        z_hat: torch.Tensor,
        max_new_tokens: int = 64,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation conditioned on plan z_hat via soft prompt."""
        device = prefix_ids.device
        B = prefix_ids.shape[0]

        # Get DiT1 output for the plan (use t=0 for clean plan)
        prefix_hidden = self._get_prefix_hidden(prefix_ids)
        t_zero = torch.zeros(B, device=device)
        h1 = self.dit1(prefix_hidden, z_hat.to(self._dtype), t_zero)
        h1_pooled = h1.mean(dim=1)
        soft_prompt = self._make_soft_prompt(h1_pooled)  # (B, K, d_lm) in bf16

        # Prepare initial input: soft_prompt tokens
        generated = prefix_ids.clone()
        past_key_values = None

        # First forward: prefix + soft prompt (all bf16)
        prefix_embeds = self.ar.get_input_embeddings()(prefix_ids)
        lm_input = torch.cat([prefix_embeds, soft_prompt], dim=1)
        out = self.ar(inputs_embeds=lm_input, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :]  # last position

        # Track generated token ids for repetition penalty
        gen_ids: list[torch.Tensor] = []

        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0 and gen_ids:
                prev = torch.cat(gen_ids, dim=-1)
                for b in range(B):
                    for tok_id in prev[b].unique():
                        if logits[b, tok_id] > 0:
                            logits[b, tok_id] /= repetition_penalty
                        else:
                            logits[b, tok_id] *= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Nucleus sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_tok_sorted = torch.multinomial(sorted_probs, 1)
            next_tok = sorted_idx.gather(-1, next_tok_sorted)

            gen_ids.append(next_tok)
            generated = torch.cat([generated, next_tok], dim=-1)

            # Next step
            next_emb = self.ar.get_input_embeddings()(next_tok)
            out = self.ar(inputs_embeds=next_emb, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]

        return generated
