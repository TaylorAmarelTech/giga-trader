"""
Centralized Logging Configuration
==================================
Sets up consistent logging format across the pipeline.
Call setup_logging() once at entry points (train_robust_model.py, giga_orchestrator.py).
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
    fmt: str = "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger with console and optional file handler.

    Parameters
    ----------
    level : int
        Logging level (default INFO).
    log_file : str, optional
        Path to log file. If None, logs to logs/giga_trader.log.
    fmt : str
        Log format string.
    datefmt : str
        Date format string.
    """
    root = logging.getLogger()

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (optional)
    if log_file is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "giga_trader.log")

    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:
        pass  # File logging is best-effort
