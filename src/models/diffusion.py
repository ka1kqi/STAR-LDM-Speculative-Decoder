"""Diffusion schedule utilities: cosine schedule, logSNR, v-prediction, sigmoid weighting."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cosine noise schedule  (Nichol & Dhariwal, 2021)
# ---------------------------------------------------------------------------

def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cumulative signal rate alpha_bar(t) under the cosine schedule.
    t in [0, 1]."""
    num = torch.cos((t + s) / (1.0 + s) * (math.pi / 2.0))
    den = math.cos(s / (1.0 + s) * (math.pi / 2.0))
    return (num / den).clamp(min=1e-8, max=1.0).pow(2)


def compute_logsnr(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """logSNR(t) = log(alpha_bar / (1 - alpha_bar))."""
    ab = cosine_alpha_bar(t, s)
    return torch.log(ab / (1.0 - ab).clamp(min=1e-8))


def sigmoid_weight(logsnr: torch.Tensor) -> torch.Tensor:
    """Loss weight = sigmoid(logSNR).  Default form; see STAR-LDM Table 3."""
    return torch.sigmoid(logsnr)


# ---------------------------------------------------------------------------
# DiffusionSchedule helper
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    """Pre-computes all noise-schedule quantities for a given batch of timesteps.

    All methods accept t in [0, 1] as a float tensor of shape (B,) or (B, 1).
    """

    def __init__(self, s: float = 0.008):
        self.s = s

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return cosine_alpha_bar(t, self.s)

    def sqrt_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar(t).sqrt()

    def sqrt_one_minus_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha_bar(t)).clamp(min=1e-8).sqrt()

    def logsnr(self, t: torch.Tensor) -> torch.Tensor:
        return compute_logsnr(t, self.s)

    # ---- forward diffusion q(z_t | z_0) ----
    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample z_t ~ q(z_t | z_0).  Returns (z_t, noise)."""
        if noise is None:
            noise = torch.randn_like(z0)
        t_ = t.view(-1, 1) if t.dim() == 1 else t
        sa = self.sqrt_alpha_bar(t_)
        sb = self.sqrt_one_minus_alpha_bar(t_)
        z_t = sa * z0 + sb * noise
        return z_t, noise

    # ---- v-prediction target ----
    def v_target(self, z0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*z0  (Salimans & Ho, 2022)."""
        t_ = t.view(-1, 1) if t.dim() == 1 else t
        sa = self.sqrt_alpha_bar(t_)
        sb = self.sqrt_one_minus_alpha_bar(t_)
        return sa * noise - sb * z0

    # ---- recover z0 from v-prediction ----
    def predict_z0(self, z_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ = t.view(-1, 1) if t.dim() == 1 else t
        sa = self.sqrt_alpha_bar(t_)
        sb = self.sqrt_one_minus_alpha_bar(t_)
        return sa * z_t - sb * v

    # ---- recover noise from v-prediction ----
    def predict_noise(self, z_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_ = t.view(-1, 1) if t.dim() == 1 else t
        sa = self.sqrt_alpha_bar(t_)
        sb = self.sqrt_one_minus_alpha_bar(t_)
        return sb * z_t + sa * v
