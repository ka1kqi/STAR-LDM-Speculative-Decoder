"""Unit tests for DDPM and DPM-Solver++ samplers."""

import pytest
import torch
from src.models.diffusion import DiffusionSchedule
from src.models.samplers import DDPMSampler, DPMSolverPPSampler


def _dummy_denoise(z_t, t, **kw):
    """Trivial denoiser that predicts zero v."""
    return torch.zeros_like(z_t)


class TestDDPMSampler:
    def test_sample_shape(self):
        schedule = DiffusionSchedule()
        sampler = DDPMSampler(schedule, num_steps=5)
        z = sampler.sample(_dummy_denoise, shape=(2, 64), device=torch.device("cpu"))
        assert z.shape == (2, 64)

    def test_deterministic_with_seed(self):
        schedule = DiffusionSchedule()
        sampler = DDPMSampler(schedule, num_steps=5)
        torch.manual_seed(0)
        z1 = sampler.sample(_dummy_denoise, shape=(1, 32), device=torch.device("cpu"))
        torch.manual_seed(0)
        z2 = sampler.sample(_dummy_denoise, shape=(1, 32), device=torch.device("cpu"))
        assert torch.allclose(z1, z2)


class TestDPMSolverPP:
    def test_sample_shape(self):
        schedule = DiffusionSchedule()
        sampler = DPMSolverPPSampler(schedule, num_steps=5)
        z = sampler.sample(_dummy_denoise, shape=(2, 64), device=torch.device("cpu"))
        assert z.shape == (2, 64)
