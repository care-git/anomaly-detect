# tests/test_platform/test_windows.py
#
# Windows-specific integration tests.  These tests run against the real Win32
# host — no sys.platform mocking.  Skip silently on macOS and Linux.

import sys

import pytest

pytestmark = pytest.mark.windows

if sys.platform != "win32":
    pytest.skip("Windows-only tests", allow_module_level=True)

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

def test_ansi_supported_returns_bool_on_windows():
    """_ansi_supported() must always return a bool, never raise on Windows."""
    result = _ansi_supported()
    assert isinstance(result, bool)


def test_ansi_supported_does_not_raise_on_windows():
    """ctypes negotiation must be exception-safe on all Windows terminal hosts."""
    try:
        _ansi_supported()
    except Exception as exc:
        pytest.fail(f"_ansi_supported() raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# GPU / DirectML
# ---------------------------------------------------------------------------

def test_setup_gpu_completes_without_error_on_windows():
    gu._GPU_CONFIGURED = False
    setup_gpu()
    assert gu._GPU_CONFIGURED is True


def test_setup_gpu_logs_wsl2_hint_when_no_cuml():
    """On Windows without cuML, users should be directed to the WSL2 path."""
    if gu._CUML_AVAILABLE:
        pytest.skip("cuML present — no-cuML branch not exercised")
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "WSL2" in all_calls or "Linux" in all_calls


@pytest.mark.gpu
def test_setup_gpu_logs_directml_when_gpu_found():
    """With a DirectX 12-capable GPU present, the log must mention DirectML."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        pytest.skip("No GPU devices found — DirectML branch not exercised")
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "DirectML" in all_calls or "DirectX" in all_calls


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_interface_comment_mentions_windows_names():
    """The config file documents Wi-Fi/Ethernet as Windows interface names."""
    with open("config/config.yml") as f:
        content = f.read()
    assert "Wi-Fi" in content or "Ethernet" in content


# ---------------------------------------------------------------------------
# Npcap
# ---------------------------------------------------------------------------

def test_scapy_import_provides_useful_npcap_message_on_missing_driver():
    """Scapy's ImportError on Windows should mention Npcap so users know how to fix it."""
    try:
        from scapy.all import sniff  # noqa: F401
    except ImportError as exc:
        msg = str(exc).lower()
        assert "npcap" in msg or "winpcap" in msg, (
            f"Expected Npcap/WinPcap mention in ImportError, got: {exc}"
        )
    except OSError as exc:
        msg = str(exc).lower()
        assert "npcap" in msg or "winpcap" in msg or "wpcap" in msg, (
            f"Expected Npcap/WinPcap mention in OSError, got: {exc}"
        )
