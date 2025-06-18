"""
logger.py
"""

import logging
from pathlib import Path
from typing import Literal

import rich.logging

LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


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
        file_handler.setFormatter(
            logging.Formatter(
                "%(name)s | %(levelname)s | %(message)s",
            )
        )
        root.addHandler(file_handler)

    root.setLevel(lvl)
