"""Unit tests for config loading and merging."""

import pytest
from pathlib import Path
from src.utils.config import load_config, merge_configs

CONFIG_DIR = Path(__file__).parent.parent / "configs"


class TestConfig:
    def test_load_default(self):
        cfg = load_config(CONFIG_DIR / "default.yaml")
        assert "model" in cfg
        assert cfg["model"]["ar_model_name"] == "Qwen/Qwen2.5-0.5B"
        assert cfg["model"]["d_plan"] == 768
        assert cfg["model"]["soft_prompt_len"] == 8
        assert cfg["model"]["beta"] == 5.0

    def test_load_toy(self):
        cfg = load_config(CONFIG_DIR / "toy.yaml")
        assert cfg["model"]["d_dit"] == 256
        assert cfg["sampler"]["num_steps"] == 5

    def test_merge_configs(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 10
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5

    def test_merge_does_not_mutate(self):
        base = {"x": {"y": 1}}
        override = {"x": {"y": 2}}
        merged = merge_configs(base, override)
        assert base["x"]["y"] == 1
        assert merged["x"]["y"] == 2
