"""
Gabriel Braun, 2025

Este módulo implementa as configurações.
"""

import configparser
import importlib
from pathlib import Path
from typing import Any, Dict

_CFG_PATH = importlib.resources.files("udgp.config").joinpath("default_config.cfg")


# Internal tables
_CFG_SOLVER = {}
_CFG_SOLVER_MODEL = {}
_CFG_SOLVER_MODEL_STAGE = {}


def _cfg_read(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
    """
    cp = configparser.ConfigParser()
    cp.read(path, encoding="utf-8")
    return {sec.lower(): dict(cp.items(sec)) for sec in cp.sections()}


def _cfg_register(sec: str, opts: Dict[str, Any]) -> None:
    """
    Regiter
    """
    parts = sec.split(".")
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
        raise ValueError(f"Invalid section name [{sec}]")


def set_config(cfg_file: str | Path) -> None:
    """
    Merge sections from *cfg_file* into the in-memory tables.
    """
    new_cfg = _cfg_read(Path(cfg_file))
    if not new_cfg:
        raise ValueError(f"{cfg_file} contained no sections")

    for sec, opts in new_cfg.items():
        _cfg_register(sec, opts)


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
for _sec, _opts in _cfg_read(_CFG_PATH).items():
    _cfg_register(_sec, _opts)
