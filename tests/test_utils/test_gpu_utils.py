# tests/test_utils/test_gpu_utils.py

from unittest.mock import patch

import pytest

import utils.gpu_utils as gu
from utils.gpu_utils import cuml_available, release_gpu_memory, setup_gpu


@pytest.fixture(autouse=True)
def reset_gpu_state():
    """Restore _GPU_CONFIGURED after each test so autoencoder tests are unaffected."""
    original = gu._GPU_CONFIGURED
    yield
    gu._GPU_CONFIGURED = original


# ---------------------------------------------------------------------------
# cuml_available
# ---------------------------------------------------------------------------

def test_cuml_available_returns_bool():
    assert isinstance(cuml_available(), bool)


def test_cuml_available_reflects_module_flag(monkeypatch):
    monkeypatch.setattr(gu, "_CUML_AVAILABLE", True)
    assert cuml_available() is True

    monkeypatch.setattr(gu, "_CUML_AVAILABLE", False)
    assert cuml_available() is False


# ---------------------------------------------------------------------------
# setup_gpu - idempotency
# ---------------------------------------------------------------------------

def test_setup_gpu_sets_configured_flag():
    gu._GPU_CONFIGURED = False
    setup_gpu()
    assert gu._GPU_CONFIGURED is True


def test_setup_gpu_is_idempotent(monkeypatch):
    """Second call must return early without touching TF or logging again."""
    gu._GPU_CONFIGURED = True
    import tensorflow as tf
    tf_calls = []
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda *a: tf_calls.append(True) or [])
    setup_gpu()
    assert len(tf_calls) == 0


def test_setup_gpu_can_be_called_multiple_times_without_error():
    gu._GPU_CONFIGURED = False
    setup_gpu()
    setup_gpu()
    setup_gpu()


# ---------------------------------------------------------------------------
# setup_gpu - platform log branches
# ---------------------------------------------------------------------------

def test_setup_gpu_logs_wsl2_hint_on_windows(monkeypatch):
    gu._GPU_CONFIGURED = False
    monkeypatch.setattr(gu.sys, "platform", "win32")
    monkeypatch.setattr(gu, "_CUML_AVAILABLE", False)

    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()

    all_info = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "WSL2" in all_info or "Linux" in all_info


def test_setup_gpu_logs_rapids_hint_on_linux(monkeypatch):
    gu._GPU_CONFIGURED = False
    monkeypatch.setattr(gu.sys, "platform", "linux")
    monkeypatch.setattr(gu, "_CUML_AVAILABLE", False)

    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()

    all_info = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "RAPIDS" in all_info or "conda" in all_info


def test_setup_gpu_logs_cuml_available_when_present(monkeypatch):
    gu._GPU_CONFIGURED = False
    monkeypatch.setattr(gu, "_CUML_AVAILABLE", True)

    with patch("utils.gpu_utils.logger") as mock_logger, \
         patch("utils.config_loader.get_config", return_value={"training": {"use_gpu": True}}):
        setup_gpu()

    all_info = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "cuML" in all_info
    assert "enabled via config" in all_info


def test_setup_gpu_logs_cuml_disabled_via_config(monkeypatch):
    gu._GPU_CONFIGURED = False
    monkeypatch.setattr(gu, "_CUML_AVAILABLE", True)

    with patch("utils.gpu_utils.logger") as mock_logger, \
         patch("utils.config_loader.get_config", return_value={"training": {"use_gpu": False}}):
        setup_gpu()

    all_info = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "cuML" in all_info
    assert "disabled via config" in all_info


# ---------------------------------------------------------------------------
# release_gpu_memory
# ---------------------------------------------------------------------------

def test_release_gpu_memory_does_not_raise():
    release_gpu_memory()


def test_release_gpu_memory_safe_when_keras_unavailable(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "keras":
            raise ImportError("keras not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    release_gpu_memory()


def test_release_gpu_memory_calls_gc_collect(monkeypatch):
    collected = []
    import gc
    monkeypatch.setattr(gc, "collect", lambda: collected.append(True))
    release_gpu_memory()
    assert len(collected) >= 1
