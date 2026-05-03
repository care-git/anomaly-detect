# tests/test_utils/test_config_loader.py

import pytest
import yaml

import utils.config_loader as cl
from utils.config_loader import get_config, set_config_path


@pytest.fixture(autouse=True)
def restore_config_state():
    """Restore module globals after every test so no test pollutes the cache."""
    original_path = cl._config_path
    original_cache = cl._config_cache
    yield
    cl._config_path = original_path
    cl._config_cache = original_cache


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

def test_get_config_returns_dict():
    config = get_config()
    assert isinstance(config, dict)


def test_get_config_contains_expected_top_level_keys():
    config = get_config()
    for key in ("capture", "preprocessing", "training", "detection", "siem"):
        assert key in config, f"Missing top-level key: {key!r}"


def test_get_config_training_has_model_type():
    config = get_config()
    assert "model_type" in config["training"]


def test_get_config_siem_has_mode():
    config = get_config()
    assert "mode" in config["siem"]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_get_config_returns_same_object_on_repeated_calls():
    first = get_config()
    second = get_config()
    assert first is second


def test_cache_populated_after_first_call():
    cl._config_cache = None
    get_config()
    assert cl._config_cache is not None


# ---------------------------------------------------------------------------
# set_config_path
# ---------------------------------------------------------------------------

def test_set_config_path_loads_custom_file(tmp_path):
    custom = tmp_path / "custom.yml"
    custom.write_text(yaml.dump({"custom_key": "custom_value"}))

    set_config_path(str(custom))
    config = get_config()

    assert config.get("custom_key") == "custom_value"


def test_set_config_path_resets_cache(tmp_path):
    get_config()
    assert cl._config_cache is not None

    custom = tmp_path / "other.yml"
    custom.write_text(yaml.dump({"new_key": 42}))

    set_config_path(str(custom))
    assert cl._config_cache is None


def test_set_config_path_same_path_does_not_reset_cache():
    get_config()
    cache_before = cl._config_cache
    set_config_path(cl._config_path)
    assert cl._config_cache is cache_before


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_get_config_raises_for_missing_file(tmp_path):
    cl._config_cache = None
    cl._config_path = str(tmp_path / "nonexistent.yml")
    with pytest.raises(FileNotFoundError):
        get_config()


def test_set_config_path_then_get_config_raises_for_missing_file(tmp_path):
    set_config_path(str(tmp_path / "ghost.yml"))
    with pytest.raises(FileNotFoundError):
        get_config()
