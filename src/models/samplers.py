"""Diffusion samplers: DDPM (baseline) and DPM-Solver++(2M) SDE.

References:
    - DDPM: Ho et al. 2020
    - DPM-Solver++: Lu et al. 2022 (arXiv 2211.01095)
"""

from __future__ import annotations

import torch
from .diffusion import DiffusionSchedule


class DDPMSampler:
    """Standard DDPM reverse-process sampler with cosine schedule.

    Parameters
    ----------
    schedule : DiffusionSchedule
    num_steps : int — number of denoising steps (default 50).
    """

    def __init__(self, schedule: DiffusionSchedule, num_steps: int = 50):
        self.schedule = schedule
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        shape: tuple[int, ...],
        device: torch.device,
        cond_kwargs: dict | None = None,
    ) -> torch.Tensor:
        """Run reverse process.

        Parameters
        ----------
        denoise_fn : callable(z_t, t, **cond_kwargs) -> v_pred
        shape : (B, D_plan)
        cond_kwargs : passed to denoise_fn each step.
        """
        if cond_kwargs is None:
            cond_kwargs = {}

        z = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        for i in range(self.num_steps):
            t_now = timesteps[i].expand(shape[0])
            t_next = timesteps[i + 1].expand(shape[0])

            v_pred = denoise_fn(z, t_now, **cond_kwargs)

            # Recover z0 and eps from v-prediction
            z0_pred = self.schedule.predict_z0(z, v_pred, t_now)
            eps_pred = self.schedule.predict_noise(z, v_pred, t_now)

            if i < self.num_steps - 1:
                # Compute posterior mean for q(z_{t-1} | z_t, z_0)
                ab_now = self.schedule.alpha_bar(t_now).view(-1, 1)
                ab_next = self.schedule.alpha_bar(t_next).view(-1, 1)

                # Simple DDPM step
                alpha_now = ab_now
                alpha_next = ab_next
                beta_tilde = (1.0 - alpha_next) / (1.0 - alpha_now) * (1.0 - alpha_now / alpha_next)
                beta_tilde = beta_tilde.clamp(min=1e-8)

                mean = (alpha_next.sqrt() * (1 - alpha_now / alpha_next) / (1 - alpha_now) * z0_pred
                        + (alpha_now / alpha_next).sqrt() * (1 - alpha_next) / (1 - alpha_now) * z)
                noise = torch.randn_like(z)
                z = mean + beta_tilde.sqrt() * noise
            else:
                z = z0_pred

        return z


class DPMSolverPPSampler:
    """DPM-Solver++(2M) SDE sampler for few-step denoising.

    A second-order multistep solver operating in the data-prediction space.

    Parameters
    ----------
    schedule : DiffusionSchedule
    num_steps : int — number of function evaluations (5–50).
    """

    def __init__(self, schedule: DiffusionSchedule, num_steps: int = 20):
        self.schedule = schedule
        self.num_steps = num_steps

    def _lambda(self, t: torch.Tensor) -> torch.Tensor:
        """λ(t) = logSNR(t) / 2."""
        return self.schedule.logsnr(t) / 2.0

    @torch.no_grad()
    def sample(
        self,
        denoise_fn,
        shape: tuple[int, ...],
        device: torch.device,
        cond_kwargs: dict | None = None,
    ) -> torch.Tensor:
        if cond_kwargs is None:
            cond_kwargs = {}

        z = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1, device=device)

        # First step: Euler (first-order DPM-Solver++)
        t0 = timesteps[0].expand(shape[0])
        v0 = denoise_fn(z, t0, **cond_kwargs)
        x0_pred_prev = self.schedule.predict_z0(z, v0, t0)

        t_next = timesteps[1].expand(shape[0])
        sa_next = self.schedule.sqrt_alpha_bar(t_next).view(-1, 1)
        sb_next = self.schedule.sqrt_one_minus_alpha_bar(t_next).view(-1, 1)
        eps0 = self.schedule.predict_noise(z, v0, t0)
        z = sa_next * x0_pred_prev + sb_next * eps0

        # Multistep (2M) iterations
        for i in range(1, self.num_steps):
            t_cur = timesteps[i].expand(shape[0])
            t_next_step = timesteps[i + 1].expand(shape[0])

            v_pred = denoise_fn(z, t_cur, **cond_kwargs)
            x0_pred = self.schedule.predict_z0(z, v_pred, t_cur)

            # Second-order correction using previous x0 prediction
            lam_cur = self._lambda(t_cur).view(-1, 1)
            lam_prev = self._lambda(timesteps[max(i - 1, 0)].expand(shape[0])).view(-1, 1)
            lam_next = self._lambda(t_next_step).view(-1, 1)

            h = lam_next - lam_cur
            h_prev = lam_cur - lam_prev
            r = h_prev / (h + 1e-8)

            # DPM-Solver++(2M) data-prediction update
            D = (1.0 + 1.0 / (2.0 * r + 1e-8)) * x0_pred - (1.0 / (2.0 * r + 1e-8)) * x0_pred_prev

            sa_n = self.schedule.sqrt_alpha_bar(t_next_step).view(-1, 1)
            sb_n = self.schedule.sqrt_one_minus_alpha_bar(t_next_step).view(-1, 1)

            eps_pred = self.schedule.predict_noise(z, v_pred, t_cur)
            z = sa_n * D + sb_n * eps_pred

            x0_pred_prev = x0_pred

        return z
