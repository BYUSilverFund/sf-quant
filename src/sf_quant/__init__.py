"""Load optional optimizer/Ray and reporting dependencies only when requested."""
from importlib import import_module

__all__ = ["backtester", "data", "optimizer", "performance", "research", "simulation"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
