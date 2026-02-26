"""Lightweight wandb wrapper — no-ops gracefully if wandb is unavailable or disabled."""

from __future__ import annotations

from typing import Any, Optional


_run = None


def init_tracking(cfg: dict, job_name: str, tags: Optional[list[str]] = None) -> bool:
    """Initialize wandb if enabled in config.  Returns True if active."""
    global _run
    track_cfg = cfg.get("tracking", {})
    if not track_cfg.get("enabled", False):
        return False

    try:
        import wandb
    except ImportError:
        return False

    _run = wandb.init(
        project=track_cfg.get("project", "diff_spec_AR_LM"),
        entity=track_cfg.get("entity"),
        name=track_cfg.get("run_name") or job_name,
        tags=tags or [],
        config=cfg,
        reinit=True,
    )
    return True


def log_metrics(metrics: dict[str, Any], step: int) -> None:
    """Log a dict of metrics at the given step."""
    if _run is None:
        return
    import wandb
    wandb.log(metrics, step=step)


def finish() -> None:
    """Finish the wandb run."""
    global _run
    if _run is None:
        return
    import wandb
    wandb.finish()
    _run = None
