"""
Gabriel Braun, 2025

Este módulo implementa as configurações.
"""

import importlib
from pathlib import Path
from typing import Any, Dict

import yaml

_CFG_PATH = importlib.resources.files("udgp.config").joinpath("default_config.yaml")

# Internal tables
_CFG_SOLVER = {}
_CFG_SOLVER_MODEL = {}
_CFG_SOLVER_MODEL_STAGE = {}


def _cfg_register(section_name: str, opts: Dict[str, Any]) -> None:
    """
    Regiter
    """
    parts = section_name.split(".")
    if len(parts) == 1:
        solver = parts[0]
        _CFG_SOLVER.setdefault(solver, {}).update(opts)
    elif len(parts) == 2:
        solver, model = parts
        _CFG_SOLVER_MODEL.setdefault((solver, model), {}).update(opts)
    elif len(parts) == 3:
        solver, model, stage = parts
        _CFG_SOLVER_MODEL_STAGE.setdefault((solver, model, stage), {}).update(opts)
    else:
        raise ValueError(f"Invalid section name [{section_name}]")


def _cfg_register_yaml(path: Path) -> None:
    """
    Returns:
    """
    cfg_data = yaml.safe_load(path.read_text())

    for section, opts in cfg_data.items():
        if not isinstance(opts, dict):
            raise ValueError(f"{section}: expected mapping, got {type(opts)}")
        _cfg_register(section, opts)


def set_config(cfg_path: str | Path) -> None:
    """
    Merge sections from *cfg_file* into the in-memory tables.
    """
    _cfg_register_yaml(cfg_path)


def get_config(
    *,
    solver: str,
    model: str | None = None,
    stage: str | None = None,
    overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Return merged options: solver → solver.model → solver.model.stage.
    """
    out = _CFG_SOLVER.get(solver.lower(), {}).copy()

    if model:
        key_sm = (solver.lower(), model.lower())
        out.update(_CFG_SOLVER_MODEL.get(key_sm, {}))
        if stage:
            key_sms = (solver.lower(), model.lower(), stage.lower())
            out.update(_CFG_SOLVER_MODEL_STAGE.get(key_sms, {}))

    if overrides:
        out.update(overrides)

    return out


# Bootstrap with built-in defaults
_cfg_register_yaml(_CFG_PATH)
