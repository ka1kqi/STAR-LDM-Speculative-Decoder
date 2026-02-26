# Lazy imports.


def __getattr__(name):
    if name == "lossless_speculative_decode":
        from .lossless import lossless_speculative_decode
        return lossless_speculative_decode
    if name == "approximate_speculative_decode":
        from .approximate import approximate_speculative_decode
        return approximate_speculative_decode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
