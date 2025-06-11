"""
Logging utilities for the multimodal_fin package.

Provides a standardized logger with timestamps and log levels.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger for a given module name.

    Args:
        name (str): Name of the logger (typically __name__ of the module).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger