"""DiT (Diffusion Transformer) blocks for STAR-LDM.

Two modules:
  - PromptEncoder (DiT1): encodes prefix representations + plan embedding.
  - DiffusionPrediction (DiT2): predicts v from noisy z_t, conditioned on DiT1 output.

Defaults from STAR-LDM Table 3 / Table 4:
  6 layers, hidden 1024, 16 heads, head_dim 64, SwiGLU FFN, adaptive RMSNorm.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with scale/shift modulated by a conditioning vector (e.g. timestep)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, mult: float = 8 / 3):
        super().__init__()
        inner = int(dim * mult)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SinusoidalTimestepEmb(nn.Module):
    """Sinusoidal timestep embedding, followed by MLP projection."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        # Cast to MLP weight dtype (e.g. bf16) before linear layers
        mlp_dtype = next(self.mlp.parameters()).dtype
        return self.mlp(emb.to(mlp_dtype))


# ---------------------------------------------------------------------------
# DiT Block
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    def __init__(self, dim: int = 1024, heads: int = 16, head_dim: int = 64,
                 cond_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm1 = AdaptiveRMSNorm(dim, cond_dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = AdaptiveRMSNorm(dim, cond_dim)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B, cond_dim)  broadcast to sequence
        c = cond.unsqueeze(1) if cond.dim() == 2 else cond
        h = self.norm1(x, c)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x, c)
        x = x + self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# PromptEncoder (DiT1)
# ---------------------------------------------------------------------------

class PromptEncoder(nn.Module):
    """DiT1: transforms prefix hidden states + plan embedding z_t into a
    conditioning sequence for DiT2 and the LM soft-prompt.

    Inputs:
        prefix_hidden : (B, Lp, D_lm)  — last-layer hidden states from the LM encoder.
        z_t           : (B, D_plan)     — noisy plan embedding.
        t             : (B,)            — diffusion timestep ∈ [0, 1].

    Returns:
        h1 : (B, Lp+1, D_dit)   — DiT1 output (the +1 comes from the plan token).
    """

    def __init__(
        self,
        d_lm: int,
        d_plan: int = 768,
        d_dit: int = 1024,
        n_layers: int = 6,
        heads: int = 16,
        head_dim: int = 64,
    ):
        super().__init__()
        self.proj_prefix = nn.Linear(d_lm, d_dit)
        self.proj_plan = nn.Linear(d_plan, d_dit)
        self.time_emb = SinusoidalTimestepEmb(d_dit)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_dit, heads, head_dim, cond_dim=d_dit) for _ in range(n_layers)]
        )
        self.norm_out = RMSNorm(d_dit)

    def forward(self, prefix_hidden: torch.Tensor, z_t: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        B = prefix_hidden.shape[0]
        h_prefix = self.proj_prefix(prefix_hidden)        # (B, Lp, D)
        h_plan = self.proj_plan(z_t).unsqueeze(1)          # (B, 1, D)
        h = torch.cat([h_prefix, h_plan], dim=1)           # (B, Lp+1, D)
        cond = self.time_emb(t)                             # (B, D)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.norm_out(h)


# ---------------------------------------------------------------------------
# DiffusionPrediction (DiT2)
# ---------------------------------------------------------------------------

class DiffusionPrediction(nn.Module):
    """DiT2: predicts v (v-prediction parameterization) from z_t conditioned
    on DiT1 output h1.

    Implements UNet-style skip connection for classifier-free guidance:
    when *null_context* is provided, it is concatenated with the h1 input
    along the feature dimension before each block (STAR-LDM appendix).

    Inputs:
        z_t  : (B, D_plan)
        h1   : (B, L, D_dit)  — from PromptEncoder
        t    : (B,)
        null_context : (B, L, D_dit) | None — null-context encoding for CFG skip

    Returns:
        v_pred : (B, D_plan)
    """

    def __init__(
        self,
        d_plan: int = 768,
        d_dit: int = 1024,
        n_layers: int = 6,
        heads: int = 16,
        head_dim: int = 64,
    ):
        super().__init__()
        self.proj_z = nn.Linear(d_plan, d_dit)
        self.time_emb = SinusoidalTimestepEmb(d_dit)
        # skip connection: concat h1 and null_context → project back
        self.skip_proj = nn.Linear(d_dit * 2, d_dit)
        self.blocks = nn.ModuleList(
            [DiTBlock(d_dit, heads, head_dim, cond_dim=d_dit) for _ in range(n_layers)]
        )
        self.norm_out = RMSNorm(d_dit)
        self.head = nn.Linear(d_dit, d_plan)

    def forward(self, z_t: torch.Tensor, h1: torch.Tensor, t: torch.Tensor,
                null_context: torch.Tensor | None = None) -> torch.Tensor:
        B = z_t.shape[0]
        hz = self.proj_z(z_t).unsqueeze(1)           # (B, 1, D)
        if null_context is not None:
            skip = self.skip_proj(torch.cat([h1, null_context], dim=-1))
        else:
            skip = h1
        seq = torch.cat([skip, hz], dim=1)            # (B, L+1, D)
        cond = self.time_emb(t)                        # (B, D)
        for blk in self.blocks:
            seq = blk(seq, cond)
        out = self.norm_out(seq[:, -1, :])             # take the z token
        return self.head(out)                          # (B, D_plan)
