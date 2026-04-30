# utils/config_loader.py

import os
import yaml
from utils.logger import get_logger

logger = get_logger(__name__, "INFO")
_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yml")
_config_cache = None


def set_config_path(path: str) -> None:
    """
    Overrides the default config path and resets the cache so the next
    get_config() call loads from the new file.

    Parameters:
        path (str): Absolute or relative path to a YAML config file.
    """
    global _config_cache, _config_path
    abs_path = os.path.abspath(path)
    if abs_path != os.path.abspath(_config_path):
        _config_path = abs_path
        _config_cache = None
        logger.info("Config path updated to: %s", abs_path)


def get_config() -> dict:
    """
    Loads and returns the parsed YAML configuration.

    Uses a global cache to prevent repeated disk reads. Raises an error if the
    configuration file cannot be found. Call set_config_path() before the first
    get_config() call to use a non-default config file.

    Returns:
        dict: Dictionary parsed from the active config file.
    """
    global _config_cache
    if _config_cache is None:
        if not os.path.exists(_config_path):
            raise FileNotFoundError(f"[!] config.yml not found at {_config_path}")

        with open(_config_path, 'r') as f:
            _config_cache = yaml.safe_load(f)

        logger.info("Configuration loaded from: %s", _config_path)

    return _config_cache
