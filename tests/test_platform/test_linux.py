# tests/test_platform/test_linux.py
#
# Linux-specific integration tests.  Guards at module level so the file is
# silently skipped on macOS and Windows hosts.

import sys

import pytest

pytestmark = pytest.mark.linux

if sys.platform != "linux":
    pytest.skip("Linux-only tests", allow_module_level=True)

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

def test_ansi_supported_returns_true_on_linux():
    assert _ansi_supported() is True


# ---------------------------------------------------------------------------
# GPU / RAPIDS
# ---------------------------------------------------------------------------

def test_setup_gpu_logs_rapids_hint_when_cuml_unavailable():
    """Without cuML, the Linux path should suggest RAPIDS/conda installation."""
    if gu._CUML_AVAILABLE:
        pytest.skip("cuML is present — RAPIDS hint branch not exercised")
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "RAPIDS" in all_calls or "conda" in all_calls


def test_setup_gpu_does_not_log_wsl2_hint_on_native_linux():
    """WSL2 hints target Windows users — they must not appear on native Linux."""
    if gu._CUML_AVAILABLE:
        pytest.skip("cuML present — no-cuML branch not exercised")
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "WSL2" not in all_calls


def test_setup_gpu_logs_cuml_message_when_available():
    """When cuML is importable the logger should confirm it is active."""
    if not gu._CUML_AVAILABLE:
        pytest.skip("cuML not installed — cuML-available branch not exercised")
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "cuML" in all_calls


def test_setup_gpu_completes_without_error_on_linux():
    gu._GPU_CONFIGURED = False
    setup_gpu()
    assert gu._GPU_CONFIGURED is True


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_config_interface_comment_mentions_eth0():
    """The config file documents eth0/ens3 as Linux default interfaces."""
    with open("config/config.yml") as f:
        content = f.read()
    assert "eth0" in content
