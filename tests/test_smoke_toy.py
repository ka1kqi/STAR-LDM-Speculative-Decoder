"""Smoke test: full toy training loop runs without error on CPU.

This test downloads Qwen2.5-0.5B on first run (~1GB).
Mark with @pytest.mark.slow if needed.
"""

import pytest
import torch
from pathlib import Path

transformers = pytest.importorskip("transformers", reason="transformers not installed")

from src.models.hybrid import HybridModel
from src.models.diffusion import DiffusionSchedule
from src.models.samplers import DDPMSampler
from src.data.dataset import ToyDataset, build_dataloader
from src.utils.config import load_config
from src.utils.seed import set_seed


@pytest.fixture
def toy_cfg():
    return load_config(Path(__file__).parent.parent / "configs" / "toy.yaml")


@pytest.mark.slow
class TestSmokeTraining:
    def test_toy_forward_pass(self, toy_cfg):
        """One forward pass of the hybrid model with toy data."""
        set_seed(42)
        model_cfg = toy_cfg["model"]

        model = HybridModel(
            ar_model_name=model_cfg["ar_model_name"],
            d_plan=model_cfg.get("d_plan", 768),
            d_dit=model_cfg.get("d_dit", 256),
            dit_layers=model_cfg.get("dit_layers", 2),
            dit_heads=model_cfg.get("dit_heads", 4),
            dit_head_dim=model_cfg.get("dit_head_dim", 64),
            soft_prompt_len=model_cfg.get("soft_prompt_len", 4),
            beta=model_cfg.get("beta", 1.0),
            gamma=0.0,
            cfg_drop_prob=0.0,
            final_noise_sigma2=0.0,
        )

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ds = ToyDataset(tokenizer, prefix_len=16, cont_len=16, num_samples=8)
        dl = build_dataloader(ds, batch_size=4)
        batch = next(iter(dl))

        outputs = model(
            prefix_ids=batch["prefix_ids"],
            cont_ids=batch["cont_ids"],
            z0=batch["z0"],
        )

        assert "loss" in outputs
        assert outputs["loss"].shape == ()
        assert not torch.isnan(outputs["loss"])

        # Backward should work
        outputs["loss"].backward()

    def test_toy_plan_and_generate(self, toy_cfg):
        """Plan + generate cycle on CPU."""
        set_seed(42)
        model_cfg = toy_cfg["model"]

        model = HybridModel(
            ar_model_name=model_cfg["ar_model_name"],
            d_plan=model_cfg.get("d_plan", 768),
            d_dit=model_cfg.get("d_dit", 256),
            dit_layers=model_cfg.get("dit_layers", 2),
            dit_heads=model_cfg.get("dit_heads", 4),
            dit_head_dim=model_cfg.get("dit_head_dim", 64),
            soft_prompt_len=model_cfg.get("soft_prompt_len", 4),
            final_noise_sigma2=0.0,
        )
        model.eval()

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["ar_model_name"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        schedule = DiffusionSchedule()
        sampler = DDPMSampler(schedule, num_steps=3)

        prefix = tokenizer.encode("Hello world", return_tensors="pt")
        z_hat = model.plan(prefix, sampler, cfg_scale=1.0)
        assert z_hat.shape == (1, 768)

        generated = model.generate_from_plan(prefix, z_hat, max_new_tokens=8)
        assert generated.shape[1] == prefix.shape[1] + 8
