# Lazy imports — avoids pulling heavy deps at collection time.


def __getattr__(name):
    if name in ("FineWebDataset", "ToyDataset", "build_dataloader"):
        from . import dataset
        return getattr(dataset, name)
    if name in ("extract_structure", "canonical_serialize"):
        from . import structure
        return getattr(structure, name)
    if name == "SentenceT5Encoder":
        from .embedding import SentenceT5Encoder
        return SentenceT5Encoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
