"""
logger.py
"""

import logging
from pathlib import Path
from typing import Literal

import rich.logging

LogLevel = Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
    "NOTSET",
]

_LEVELS = {n: getattr(logging, n) for n in logging.getLevelNamesMapping()}


def _resolve_level(level: str | int) -> int:
    """
    Returns: (int) logging level from str representation.
    """
    if isinstance(level, str):
        try:
            return _LEVELS[level.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid log level: {level!r}") from exc
    if isinstance(level, int):
        return level
    raise TypeError("level must be str or int")


class _SolverInfoFilter(logging.Filter):
    """
    Filter solver info.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # GUROBI/GUROBIPY
        if record.name.startswith("gurobi"):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


def set_logger(
    level: LogLevel | int = "WARNING",
    *,
    log_file: str | Path | None = None,
    log_to_console: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Configure global logging for UDGP.
    """
    lvl = _resolve_level(level)

    # 1) wipe previous config (idempotent re-call)
    root = logging.getLogger()
    root.handlers.clear()

    solver_info_filter = _SolverInfoFilter()

    # 2) console handler (only if requested)
    if log_to_console:
        console = rich.logging.RichHandler(
            markup=True,
            rich_tracebacks=True,
            show_level=False,
            show_time=False,
            show_path=False,
        )
        console.setLevel(lvl)
        console.addFilter(solver_info_filter)
        console.setFormatter(
            logging.Formatter(
                "%(message)s",
            )
        )
        root.addHandler(console)

    # 3) optional file handler
    if log_file is not None:
        mode = "w" if overwrite else "a"
        file_handler = logging.FileHandler(
            log_file,
            mode=mode,
            encoding="utf-8",
        )
        file_handler.setLevel(lvl)
        file_handler.addFilter(solver_info_filter)
        file_handler.setFormatter(
            logging.Formatter(
                "%(message)s",
                # "[ %(levelname)s ] %(message)s",
                # "%(name)s | %(levelname)s | %(message)s",
            )
        )
        root.addHandler(file_handler)

    root.setLevel(lvl)
