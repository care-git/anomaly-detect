# utils/logger.py

import logging
import sys

def get_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
    """
    Creates and returns a module-specific logger with standard formatting.

    Parameters:
        name (str): Logger name, usually '__name__'.
        level (str): Logging level (e.g., 'DEBUG', 'INFO', or 'WARNING').

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
