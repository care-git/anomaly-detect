# tests/test_platform/test_macos.py
#
# macOS-specific integration tests.  These tests run against the real OS — no
# mocking of sys.platform — so they are guarded by a skipif condition and only
# execute on darwin hosts.  Unit-level mocking of platform branches lives in
# tests/test_utils/.

import sys

import pytest

pytestmark = pytest.mark.macos

if sys.platform != "darwin":
    pytest.skip("macOS-only tests", allow_module_level=True)

import utils.gpu_utils as gu
from utils.gpu_utils import setup_gpu
from utils.progress import _ansi_supported


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_gpu_state():
    original = gu._GPU_CONFIGURED
    yield
    gu._GPU_CONFIGURED = original


# ---------------------------------------------------------------------------
# ANSI support
# ---------------------------------------------------------------------------

def test_ansi_supported_returns_true_on_macos():
    """macOS terminals always support ANSI — no ctypes negotiation needed."""
    assert _ansi_supported() is True


# ---------------------------------------------------------------------------
# GPU / Metal
# ---------------------------------------------------------------------------

def test_setup_gpu_does_not_log_directml_on_macos(monkeypatch):
    """DirectML is a Windows-only acceleration path and must never appear on macOS."""
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "DirectML" not in all_calls
    assert "DirectX" not in all_calls


def test_setup_gpu_does_not_log_wsl2_on_macos(monkeypatch):
    """WSL2 hints are Windows-only and must not appear in macOS log output."""
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "WSL2" not in all_calls


def test_setup_gpu_completes_without_error_on_macos():
    """setup_gpu() must not raise on a real macOS host with Metal available."""
    gu._GPU_CONFIGURED = False
    setup_gpu()
    assert gu._GPU_CONFIGURED is True


@pytest.mark.gpu
def test_metal_gpu_detected_on_apple_silicon():
    """On Apple Silicon with tensorflow-metal installed, at least one GPU is found."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) >= 1, "Expected Metal GPU device — is tensorflow-metal installed?"


@pytest.mark.gpu
def test_setup_gpu_logs_gpu_device_name_on_macos():
    """With Metal present, setup_gpu() should log the device name (not a CPU fallback)."""
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "GPU" in all_calls
    assert "no GPU found" not in all_calls


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_interface_comment_mentions_en0():
    """The config file documents en0 as the macOS default interface."""
    config_path = "config/config.yml"
    with open(config_path) as f:
        content = f.read()
    assert "en0" in content
