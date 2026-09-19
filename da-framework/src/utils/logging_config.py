"""
Structured logging configuration for the DA framework pipeline.

Provides consistent, color-coded console output and rotating file logs
for all pipeline phases.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


# ---------------------------------------------------------------------------
# Colour codes for console output (Windows-compatible via ANSI)
# ---------------------------------------------------------------------------
_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colour codes to the log level name."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _COLORS["RESET"])
        record.levelname = f"{color}{record.levelname:<8}{_COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = "da",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Create (or retrieve) a logger with console + file handlers.

    Parameters
    ----------
    name : str
        Logger name.  Use dotted names for sub-loggers
        (e.g. ``da.model``).
    log_dir : str | Path
        Directory where log files are stored.
    level : int
        Minimum severity level for the root logger.
    max_bytes : int
        Maximum size per log file before rotation.
    backup_count : int
        Number of rotated log files to keep.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ---- Console handler (coloured) ----
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        _ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    # ---- File handler (rotating) ----
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / f"{name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
    )
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "da") -> logging.Logger:
    """Convenience shortcut to retrieve an existing logger."""
    return logging.getLogger(name)
