"""
Central registry for uDGP model classes.
"""

from .gurobipy import _GPY_MODELS
from .pyomo import _PYO_MODELS

_BACKENDS = {
    "pyomo": _PYO_MODELS,
    "gurobipy": _GPY_MODELS,
}


def get_model(model_name: str, *, backend: str = "pyomo"):
    """
    Return the *class* that implements ``tag`` on the selected *backend*.
    """
    model_name = model_name.upper()
    backend_low = backend.lower()

    try:
        registry = _BACKENDS[backend_low]
    except KeyError as exc:  # unknown backend
        raise ValueError(f"Unknown backend: '{backend}'") from exc

    try:
        return registry[model_name]
    except KeyError as exc:  # unknown model tag
        raise ValueError(
            f"Unknown model '{model_name}' for backend '{backend}'"
        ) from exc
