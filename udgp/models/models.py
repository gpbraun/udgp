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
    model_name = model_name.upper()
    backend_low = backend.lower()

    try:
        model_registry = _BACKENDS[backend_low]
    except KeyError as exc:  # unknown backend
        raise ValueError(f"Unknown backend: '{backend}'") from exc

    try:
        return model_registry[model_name](**kwargs)
    except KeyError as exc:  # unknown model tag
        raise ValueError(
            f"Unknown model '{model_name}' for backend '{backend}'"
        ) from exc
