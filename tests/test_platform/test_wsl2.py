# tests/test_platform/test_wsl2.py
#
# WSL2-specific integration tests.  WSL2 presents as Linux (sys.platform ==
# "linux") but its kernel uname release string contains "microsoft", and the
# WSL_DISTRO_NAME environment variable is set.  Both signals are checked so the
# skip condition matches either indicator.

import os
import platform
import sys

import pytest

pytestmark = pytest.mark.wsl2


def _running_on_wsl2() -> bool:
    if sys.platform != "linux":
        return False
    release = platform.uname().release.lower()
    if "microsoft" in release:
        return True
    return "WSL_DISTRO_NAME" in os.environ


if not _running_on_wsl2():
    pytest.skip("WSL2-only tests", allow_module_level=True)

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
# WSL2 detection
# ---------------------------------------------------------------------------

def test_wsl2_detected_via_uname_or_env():
    """At least one of the two WSL2 detection signals must be present."""
    release = platform.uname().release.lower()
    uname_signal = "microsoft" in release
    env_signal = "WSL_DISTRO_NAME" in os.environ
    assert uname_signal or env_signal, (
        "Expected 'microsoft' in kernel release or WSL_DISTRO_NAME env var"
    )


def test_wsl2_uname_contains_microsoft_or_env_var_set():
    release = platform.uname().release.lower()
    assert "microsoft" in release or "WSL_DISTRO_NAME" in os.environ


def test_sys_platform_is_linux_under_wsl2():
    """WSL2 must report as linux, not win32 — ensures Linux code paths are taken."""
    assert sys.platform == "linux"


# ---------------------------------------------------------------------------
# ANSI support — WSL2 takes the Linux path (True unconditionally)
# ---------------------------------------------------------------------------

def test_ansi_supported_returns_true_under_wsl2():
    assert _ansi_supported() is True


# ---------------------------------------------------------------------------
# GPU — WSL2 uses the Linux/RAPIDS path
# ---------------------------------------------------------------------------

def test_setup_gpu_logs_rapids_hint_under_wsl2_when_cuml_unavailable():
    """Without cuML, WSL2 should suggest the Linux/RAPIDS installation path."""
    if gu._CUML_AVAILABLE:
        pytest.skip("cuML present — no-cuML branch not exercised")
    gu._GPU_CONFIGURED = False
    from unittest.mock import patch
    with patch("utils.gpu_utils.logger") as mock_logger:
        setup_gpu()
    all_calls = " ".join(str(c) for c in mock_logger.info.call_args_list)
    assert "RAPIDS" in all_calls or "conda" in all_calls


def test_setup_gpu_completes_without_error_under_wsl2():
    gu._GPU_CONFIGURED = False
    setup_gpu()
    assert gu._GPU_CONFIGURED is True


@pytest.mark.gpu
def test_cuda_gpu_visible_under_wsl2():
    """On WSL2 with CUDA passthrough enabled, at least one GPU should be found."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    assert len(gpus) >= 1, (
        "No GPU found under WSL2 — is CUDA passthrough configured? "
        "See README for WSL2 GPU setup instructions."
    )
