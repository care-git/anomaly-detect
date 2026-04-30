# utils/config_loader.py

import os
import yaml
from utils.logger import get_logger

logger = get_logger(__name__, "INFO")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yml")
_config_cache = None

def get_config():
    """
    Loads and returns the parsed YAML configuration.

    Uses a global cache to prevent repeated disk reads. Raises an error if the 
    configuration file cannot be found.

    Returns:
        dict: Dictionary parsed from `config/config.yml`.
    """
    global _config_cache
    if _config_cache is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"[!] config.yml not found at {CONFIG_PATH}")
        
        with open(CONFIG_PATH, 'r') as f:
            _config_cache = yaml.safe_load(f)

        logger.info("Configuration loaded from: %s", CONFIG_PATH)
        
    return _config_cache
