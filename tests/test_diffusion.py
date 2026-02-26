"""Unit tests for diffusion schedule, v-prediction, and loss weighting."""

import pytest
import torch
from src.models.diffusion import (
    cosine_alpha_bar,
    compute_logsnr,
    sigmoid_weight,
    DiffusionSchedule,
)


class TestCosineSchedule:
    def test_boundary_values(self):
        t = torch.tensor([0.0, 1.0])
        ab = cosine_alpha_bar(t)
        assert ab[0].item() > 0.99  # near 1 at t=0
        assert ab[1].item() < 0.01  # near 0 at t=1

    def test_monotonically_decreasing(self):
        t = torch.linspace(0, 1, 100)
        ab = cosine_alpha_bar(t)
        diffs = ab[1:] - ab[:-1]
        assert (diffs <= 0).all()


class TestLogSNR:
    def test_high_at_t0(self):
        logsnr = compute_logsnr(torch.tensor([0.01]))
        assert logsnr.item() > 5.0

    def test_low_at_t1(self):
        logsnr = compute_logsnr(torch.tensor([0.99]))
        assert logsnr.item() < -5.0


class TestSigmoidWeight:
    def test_range(self):
        logsnr = torch.tensor([-10.0, 0.0, 10.0])
        w = sigmoid_weight(logsnr)
        assert (w >= 0).all()
        assert (w <= 1).all()

    def test_symmetry(self):
        w_pos = sigmoid_weight(torch.tensor([5.0]))
        w_neg = sigmoid_weight(torch.tensor([-5.0]))
        assert abs(w_pos.item() + w_neg.item() - 1.0) < 0.01


class TestDiffusionSchedule:
    @pytest.fixture
    def schedule(self):
        return DiffusionSchedule()

    def test_q_sample_shape(self, schedule):
        z0 = torch.randn(4, 768)
        t = torch.rand(4)
        z_t, noise = schedule.q_sample(z0, t)
        assert z_t.shape == (4, 768)
        assert noise.shape == (4, 768)

    def test_v_target_and_recovery(self, schedule):
        z0 = torch.randn(4, 768)
        t = torch.rand(4)
        z_t, noise = schedule.q_sample(z0, t)
        v = schedule.v_target(z0, noise, t)

        z0_rec = schedule.predict_z0(z_t, v, t)
        assert torch.allclose(z0, z0_rec, atol=1e-4)

    def test_noise_recovery(self, schedule):
        z0 = torch.randn(4, 768)
        t = torch.rand(4)
        noise = torch.randn_like(z0)
        z_t, _ = schedule.q_sample(z0, t, noise=noise)
        v = schedule.v_target(z0, noise, t)

        noise_rec = schedule.predict_noise(z_t, v, t)
        assert torch.allclose(noise, noise_rec, atol=1e-4)
