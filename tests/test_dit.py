"""Unit tests for DiT blocks, PromptEncoder, and DiffusionPrediction."""

import pytest
import torch
from src.models.dit import (
    RMSNorm,
    AdaptiveRMSNorm,
    SwiGLUFFN,
    SinusoidalTimestepEmb,
    DiTBlock,
    PromptEncoder,
    DiffusionPrediction,
)


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)


class TestSwiGLU:
    def test_output_shape(self):
        ffn = SwiGLUFFN(64)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == (2, 10, 64)


class TestTimestepEmb:
    def test_output_shape(self):
        emb = SinusoidalTimestepEmb(128)
        t = torch.rand(4)
        out = emb(t)
        assert out.shape == (4, 128)


class TestDiTBlock:
    def test_forward(self):
        block = DiTBlock(dim=64, heads=4, head_dim=16, cond_dim=64)
        x = torch.randn(2, 8, 64)
        cond = torch.randn(2, 64)
        out = block(x, cond)
        assert out.shape == (2, 8, 64)


class TestPromptEncoder:
    def test_forward(self):
        d_lm = 128
        d_plan = 64
        d_dit = 64
        enc = PromptEncoder(d_lm=d_lm, d_plan=d_plan, d_dit=d_dit,
                            n_layers=2, heads=4, head_dim=16)
        prefix_hidden = torch.randn(2, 8, d_lm)
        z_t = torch.randn(2, d_plan)
        t = torch.rand(2)
        out = enc(prefix_hidden, z_t, t)
        assert out.shape == (2, 9, d_dit)  # 8 prefix + 1 plan token


class TestDiffusionPrediction:
    def test_forward(self):
        d_plan = 64
        d_dit = 64
        pred = DiffusionPrediction(d_plan=d_plan, d_dit=d_dit,
                                   n_layers=2, heads=4, head_dim=16)
        z_t = torch.randn(2, d_plan)
        h1 = torch.randn(2, 9, d_dit)
        t = torch.rand(2)
        v = pred(z_t, h1, t)
        assert v.shape == (2, d_plan)

    def test_with_null_context(self):
        d_plan = 64
        d_dit = 64
        pred = DiffusionPrediction(d_plan=d_plan, d_dit=d_dit,
                                   n_layers=2, heads=4, head_dim=16)
        z_t = torch.randn(2, d_plan)
        h1 = torch.randn(2, 9, d_dit)
        null_ctx = torch.randn(2, 9, d_dit)
        t = torch.rand(2)
        v = pred(z_t, h1, t, null_context=null_ctx)
        assert v.shape == (2, d_plan)
