"""Noise-conditioned classifier MLP for classifier guidance on structure attributes.

Architecture from STAR-LDM Table 4:
    768 → 1536, 4 residual blocks, sinusoidal time embedding,
    AdamW lr 1e-4, batch 256.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dit import SinusoidalTimestepEmb


class ResBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x + self.cond_proj(cond))
        h = self.fc2(F.silu(self.fc1(h)))
        return x + h


class GuidanceMLP(nn.Module):
    """Noise-conditioned MLP classifier for structure attributes.

    Outputs logits for each attribute head.

    Parameters
    ----------
    d_plan : int  — dimension of z_t input (768).
    d_hidden : int  — hidden dimension (1536 per Table 4).
    n_blocks : int  — number of residual blocks (4).
    attribute_dims : dict[str, int]
        Mapping from attribute name to number of classes.
        Default: {"length_bucket": 4, "has_list": 2, "has_question": 2}.
    """

    def __init__(
        self,
        d_plan: int = 768,
        d_hidden: int = 1536,
        n_blocks: int = 4,
        attribute_dims: dict[str, int] | None = None,
    ):
        super().__init__()
        if attribute_dims is None:
            attribute_dims = {
                "length_bucket": 4,
                "has_list": 2,
                "has_question": 2,
            }
        self.attribute_dims = attribute_dims
        self.proj_in = nn.Linear(d_plan, d_hidden)
        self.time_emb = SinusoidalTimestepEmb(d_hidden)
        self.blocks = nn.ModuleList(
            [ResBlock(d_hidden, d_hidden) for _ in range(n_blocks)]
        )
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_hidden, n_cls) for name, n_cls in attribute_dims.items()}
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict of logits per attribute head."""
        h = self.proj_in(z_t)
        cond = self.time_emb(t)
        for blk in self.blocks:
            h = blk(h, cond)
        return {name: head(h) for name, head in self.heads.items()}

    def compute_loss(self, z_t: torch.Tensor, t: torch.Tensor,
                     labels: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(z_t, t)
        loss = torch.tensor(0.0, device=z_t.device)
        for name, lg in logits.items():
            if name in labels:
                loss = loss + F.cross_entropy(lg, labels[name])
        return loss

    def classifier_gradient(self, z_t: torch.Tensor, t: torch.Tensor,
                            target_labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute ∇_{z_t} log p(y | z_t, t) for classifier guidance."""
        z_t = z_t.detach().requires_grad_(True)
        logits = self.forward(z_t, t)
        log_prob = torch.tensor(0.0, device=z_t.device)
        for name, lg in logits.items():
            if name in target_labels:
                log_prob = log_prob + F.log_softmax(lg, dim=-1).gather(
                    1, target_labels[name].unsqueeze(-1)
                ).sum()
        log_prob.backward()
        assert z_t.grad is not None
        return z_t.grad.detach()
