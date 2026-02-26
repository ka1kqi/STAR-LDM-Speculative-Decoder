# Lazy imports.


def __getattr__(name):
    if name == "evaluate_c4":
        from .c4_eval import evaluate_c4
        return evaluate_c4
    if name == "evaluate_storycloze":
        from .storycloze import evaluate_storycloze
        return evaluate_storycloze
    if name == "make_eval_plots":
        from .plots import make_eval_plots
        return make_eval_plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
