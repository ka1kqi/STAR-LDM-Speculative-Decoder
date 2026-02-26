"""Unit tests for the guidance MLP classifier."""

import pytest
import torch
from src.models.guidance_mlp import GuidanceMLP


class TestGuidanceMLP:
    @pytest.fixture
    def mlp(self):
        return GuidanceMLP(d_plan=64, d_hidden=128, n_blocks=2)

    def test_forward_shape(self, mlp):
        z_t = torch.randn(4, 64)
        t = torch.rand(4)
        logits = mlp(z_t, t)
        assert "length_bucket" in logits
        assert logits["length_bucket"].shape == (4, 4)
        assert logits["has_list"].shape == (4, 2)

    def test_compute_loss(self, mlp):
        z_t = torch.randn(4, 64)
        t = torch.rand(4)
        labels = {
            "length_bucket": torch.randint(0, 4, (4,)),
            "has_list": torch.randint(0, 2, (4,)),
            "has_question": torch.randint(0, 2, (4,)),
        }
        loss = mlp.compute_loss(z_t, t, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_classifier_gradient(self, mlp):
        z_t = torch.randn(4, 64)
        t = torch.rand(4)
        labels = {"length_bucket": torch.tensor([0, 1, 2, 3])}
        grad = mlp.classifier_gradient(z_t, t, labels)
        assert grad.shape == (4, 64)
        assert not torch.isnan(grad).any()
