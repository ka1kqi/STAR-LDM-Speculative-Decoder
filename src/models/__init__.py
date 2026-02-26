# Lazy imports to avoid pulling in transformers at collection time.
# Users should import directly from submodules, e.g.:
#   from src.models.diffusion import DiffusionSchedule
#   from src.models.hybrid import HybridModel


def __getattr__(name):
    if name == "DiTBlock":
        from .dit import DiTBlock
        return DiTBlock
    if name == "PromptEncoder":
        from .dit import PromptEncoder
        return PromptEncoder
    if name == "DiffusionPrediction":
        from .dit import DiffusionPrediction
        return DiffusionPrediction
    if name == "DiffusionSchedule":
        from .diffusion import DiffusionSchedule
        return DiffusionSchedule
    if name in ("compute_logsnr", "sigmoid_weight"):
        from . import diffusion
        return getattr(diffusion, name)
    if name == "HybridModel":
        from .hybrid import HybridModel
        return HybridModel
    if name == "GuidanceMLP":
        from .guidance_mlp import GuidanceMLP
        return GuidanceMLP
    if name in ("DDPMSampler", "DPMSolverPPSampler"):
        from . import samplers
        return getattr(samplers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
