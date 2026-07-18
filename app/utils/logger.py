"""
BAYMAX AI – Structured Logger
==============================
Provides a pre-configured `loguru` logger with:
- Colored, human-readable console output in development
- JSON-structured file logging for production
- Automatic log rotation and retention

Usage:
    from app.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Module loaded")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# ─── Remove default loguru handler ───────────────────────────────────────────
logger.remove()


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    json_logs: bool = False,
) -> None:
    """
    Configure the global loguru logger.

    Args:
        log_level:  Minimum severity level to capture (e.g. "DEBUG", "INFO").
        log_dir:    Directory path for log file output. No file logging if None.
        json_logs:  If True, serialize log records as JSON (for production).
    """
    fmt_console = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> – "
        "<level>{message}</level>"
    )

    fmt_file = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} – {message}"
    )

    # Console handler
    logger.add(
        sys.stderr,
        format=fmt_console,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "baymax_{time:YYYY-MM-DD}.log"

        logger.add(
            str(log_path),
            format=fmt_file if not json_logs else "{message}",
            level=log_level,
            rotation="100 MB",
            retention="30 days",
            compression="zip",
            backtrace=True,
            diagnose=False,
            serialize=json_logs,
            enqueue=True,  # Thread-safe async logging
        )

    logger.info(
        "BAYMAX logging initialized | level={} | file_logging={}",
        log_level,
        log_dir is not None,
    )


def get_logger(name: str) -> "Logger":
    """
    Return a loguru logger bound to the given module name.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A loguru Logger instance.
    """
    return logger.bind(name=name)
