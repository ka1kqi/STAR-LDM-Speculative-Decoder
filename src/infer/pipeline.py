"""Unified inference pipeline with toggles for sampler, guidance, and speculation."""

from __future__ import annotations

import torch
from typing import Optional

from ..models.hybrid import HybridModel
from ..models.samplers import DDPMSampler, DPMSolverPPSampler
from ..models.diffusion import DiffusionSchedule, compute_logsnr
from ..models.guidance_mlp import GuidanceMLP
from ..spec_decode.lossless import lossless_speculative_decode
from ..spec_decode.approximate import approximate_speculative_decode


class InferencePipeline:
    """End-to-end inference: plan → (optionally speculative) generate.

    Parameters
    ----------
    hybrid_model : HybridModel
    sampler_type : "ddpm" or "dpm_solver_pp"
    sampler_steps : number of diffusion steps
    cfg_scale : classifier-free guidance scale (1.0 = no guidance)
    guidance_mlp : optional GuidanceMLP for classifier guidance
    guidance_labels : dict[str, Tensor] target labels for classifier guidance
    guidance_scale : classifier guidance strength
    spec_mode : None, "lossless", or "approximate"
    draft_model : AR draft model for speculative decoding
    draft_k : number of draft tokens per speculation block
    top_p : nucleus sampling p
    temperature : sampling temperature
    repetition_penalty : repetition penalty factor
    max_new_tokens : maximum continuation tokens
    tau_min, tau_max, tau_scale : approximate spec-decode τ(c) parameters
    """

    def __init__(
        self,
        hybrid_model: HybridModel,
        sampler_type: str = "ddpm",
        sampler_steps: int = 50,
        cfg_scale: float = 1.0,
        guidance_mlp: Optional[GuidanceMLP] = None,
        guidance_labels: Optional[dict[str, torch.Tensor]] = None,
        guidance_scale: float = 0.0,
        spec_mode: Optional[str] = None,
        draft_model=None,
        draft_k: int = 4,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 64,
        tau_min: float = 0.3,
        tau_max: float = 0.95,
        tau_scale: float = 1.0,
    ):
        self.model = hybrid_model
        self.cfg_scale = cfg_scale
        self.guidance_mlp = guidance_mlp
        self.guidance_labels = guidance_labels
        self.guidance_scale = guidance_scale
        self.spec_mode = spec_mode
        self.draft_model = draft_model
        self.draft_k = draft_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_scale = tau_scale

        schedule = DiffusionSchedule()
        if sampler_type == "dpm_solver_pp":
            self.sampler = DPMSolverPPSampler(schedule, num_steps=sampler_steps)
        else:
            self.sampler = DDPMSampler(schedule, num_steps=sampler_steps)

    @torch.no_grad()
    def __call__(self, prefix_ids: torch.Tensor) -> dict:
        """Run full inference pipeline.

        Returns dict with:
            generated_ids : full output token sequence
            z_hat : plan embedding
            spec_stats : speculative decoding stats (if applicable)
        """
        device = prefix_ids.device

        # Step 1: THINK — diffusion planning
        z_hat = self.model.plan(
            prefix_ids,
            sampler=self.sampler,
            cfg_scale=self.cfg_scale,
            guidance_mlp=self.guidance_mlp,
            guidance_labels=self.guidance_labels,
            guidance_scale=self.guidance_scale,
        )

        # Step 2: TALK — autoregressive generation
        spec_stats = {}

        if self.spec_mode and self.draft_model is not None:
            # Compute soft prompt for speculative decoding
            B = prefix_ids.shape[0]
            prefix_hidden = self.model._get_prefix_hidden(prefix_ids)
            t_zero = torch.zeros(B, device=device)
            h1 = self.model.dit1(prefix_hidden, z_hat, t_zero)
            h1_pooled = h1.mean(dim=1)
            soft_prompt = self.model._make_soft_prompt(h1_pooled)

            if self.spec_mode == "lossless":
                result = lossless_speculative_decode(
                    target_model=self.model.ar,
                    draft_model=self.draft_model,
                    input_ids=prefix_ids,
                    max_new_tokens=self.max_new_tokens,
                    draft_k=self.draft_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    repetition_penalty=self.repetition_penalty,
                    soft_prompt_embeds=soft_prompt,
                )
            elif self.spec_mode == "approximate":
                # Estimate plan logSNR from z_hat norm (heuristic; one-time GPU sync)
                plan_logsnr = float(z_hat.norm(dim=-1).mean().cpu())
                result = approximate_speculative_decode(
                    target_model=self.model.ar,
                    draft_model=self.draft_model,
                    input_ids=prefix_ids,
                    plan_logsnr=plan_logsnr,
                    max_new_tokens=self.max_new_tokens,
                    draft_k=self.draft_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    tau_min=self.tau_min,
                    tau_max=self.tau_max,
                    tau_scale=self.tau_scale,
                    soft_prompt_embeds=soft_prompt,
                )
            else:
                raise ValueError(f"Unknown spec_mode: {self.spec_mode}")

            generated_ids = result["generated_ids"]
            spec_stats = {k: v for k, v in result.items() if k != "generated_ids"}
        else:
            # Standard generation (no speculation)
            generated_ids = self.model.generate_from_plan(
                prefix_ids=prefix_ids,
                z_hat=z_hat,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
            )

        return {
            "generated_ids": generated_ids,
            "z_hat": z_hat,
            "spec_stats": spec_stats,
        }
