"""
Central registry for uDGP model classes.
"""

from .gurobipy import _GP_MODELS
from .pyomo import _PO_MODELS

_BACKENDS = {
    "pyomo": _PO_MODELS,
    "gurobipy": _GP_MODELS,
}


def get_model(model_name: str, *, backend: str = "pyomo", **kwargs):
    """
    Return the *class* that implements ``tag`` on the selected *backend*.
    """
    try:
        model_registry = _BACKENDS[backend.lower()]
    except KeyError:
        raise ValueError(f"Unknown backend: '{backend}'")

    try:
        model = model_registry[model_name.upper()]
    except KeyError:
        raise ValueError(f"Unknown model '{model_name}' for backend '{backend}'")

    return model(**kwargs)
