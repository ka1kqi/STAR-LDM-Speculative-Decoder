# Lazy imports.


def __getattr__(name):
    if name == "InferencePipeline":
        from .pipeline import InferencePipeline
        return InferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
