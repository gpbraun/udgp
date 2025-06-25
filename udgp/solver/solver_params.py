"""
Gabriel Braun, 2025

Este módulo implementa as configurações.
"""

from importlib import resources
from pathlib import Path
from typing import Any, Dict

import yaml

_PARAMS_DEFAULTS_PATH = resources.files("udgp.solver").joinpath("solver_params.yaml")

# Internal tables
_PARAMS_SOLVER = {}
_PARAMS_SOLVER_MODEL = {}
_PARAMS_SOLVER_MODEL_STAGE = {}


def _cfg_register(section_name: str, opts: Dict[str, Any]) -> None:
    """
    Regiter
    """
    parts = section_name.lower().split(".")
    if len(parts) == 1:
        solver = parts[0]
        _PARAMS_SOLVER.setdefault(solver, {}).update(opts)
    elif len(parts) == 2:
        solver, model = parts
        _PARAMS_SOLVER_MODEL.setdefault((solver, model), {}).update(opts)
    elif len(parts) == 3:
        solver, model, stage = parts
        _PARAMS_SOLVER_MODEL_STAGE.setdefault((solver, model, stage), {}).update(opts)
    else:
        raise ValueError(f"Invalid section name [{section_name}]")


def set_solver_params(
    config_path: str | Path,
) -> None:
    """
    Merge sections from *cfg_file* into the in-memory tables.
    """
    cfg_data = yaml.safe_load(Path(config_path).read_text())

    for section, opts in cfg_data.items():
        if not isinstance(opts, dict):
            raise ValueError(f"{section}: expected mapping, got {type(opts)}")
        _cfg_register(section, opts)


def get_solver_params(
    solver: str,
    model: str | None = None,
    stage: str | None = None,
) -> Dict[str, Any]:
    """
    Return merged options: solver → solver.model → solver.model.stage.
    """
    out = _PARAMS_SOLVER.get(solver.lower(), {}).copy()

    if model:
        key_sm = (solver.lower(), model.lower())
        out.update(_PARAMS_SOLVER_MODEL.get(key_sm, {}))
        if stage:
            key_sms = (solver.lower(), model.lower(), stage.lower())
            out.update(_PARAMS_SOLVER_MODEL_STAGE.get(key_sms, {}))

    return out


# Bootstrap with built-in defaults
set_solver_params(_PARAMS_DEFAULTS_PATH)
