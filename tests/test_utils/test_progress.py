# tests/test_utils/test_progress.py

import sys
import time
import pytest

from utils.progress import TrainingSpinner, _ansi_supported, tqdm_bar


# ---------------------------------------------------------------------------
# tqdm_bar (existing coverage kept)
# ---------------------------------------------------------------------------

def test_tqdm_bar_preserves_all_items():
    data = list(range(5))
    output = [i for i in tqdm_bar(data, desc="test", leave=False)]
    assert output == data


def test_tqdm_bar_disabled_still_iterates():
    data = list(range(3))
    output = [i for i in tqdm_bar(data, disable=True)]
    assert output == data


# ---------------------------------------------------------------------------
# _ansi_supported
# ---------------------------------------------------------------------------

def test_ansi_supported_returns_bool():
    assert isinstance(_ansi_supported(), bool)


def test_ansi_supported_true_on_non_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert _ansi_supported() is True


def test_ansi_supported_true_on_darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _ansi_supported() is True


def test_ansi_supported_false_on_windows_when_ctypes_unavailable(monkeypatch):
    """Simulate a Windows host where the ctypes import itself fails."""
    monkeypatch.setattr(sys, "platform", "win32")
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("ctypes", "ctypes.wintypes"):
            raise ImportError("ctypes unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    assert _ansi_supported() is False


# ---------------------------------------------------------------------------
# TrainingSpinner — frame / mark selection
# ---------------------------------------------------------------------------

def test_spinner_uses_ansi_frames_when_ansi_enabled(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", True)
    spinner = TrainingSpinner("Test")
    assert spinner._frames == TrainingSpinner._FRAMES_ANSI
    assert spinner._marks == TrainingSpinner._MARKS_ANSI


def test_spinner_uses_ascii_frames_when_ansi_disabled(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("Test")
    assert spinner._frames == TrainingSpinner._FRAMES_ASCII
    assert spinner._marks == TrainingSpinner._MARKS_ASCII


# ---------------------------------------------------------------------------
# TrainingSpinner — start / stop lifecycle
# ---------------------------------------------------------------------------

def test_spinner_context_manager_completes_without_error(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", False)
    with TrainingSpinner("Task"):
        pass


def test_spinner_context_manager_handles_exception(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", False)
    with pytest.raises(ValueError):
        with TrainingSpinner("Task"):
            raise ValueError("expected")


def test_spinner_stop_success_writes_done_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("MyTask")
    spinner.start()
    spinner.stop(success=True)
    err = capsys.readouterr().err
    assert "done" in err
    assert "MyTask" in err


def test_spinner_stop_failure_writes_failed_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("MyTask")
    spinner.start()
    spinner.stop(success=False)
    err = capsys.readouterr().err
    assert "FAILED" in err
    assert "MyTask" in err


def test_spinner_stop_success_ansi_writes_checkmark(monkeypatch, capsys):
    monkeypatch.setattr("utils.progress._ANSI", True)
    spinner = TrainingSpinner("MyTask")
    spinner.start()
    spinner.stop(success=True)
    err = capsys.readouterr().err
    assert "✓" in err


def test_spinner_stop_failure_ansi_writes_cross(monkeypatch, capsys):
    monkeypatch.setattr("utils.progress._ANSI", True)
    spinner = TrainingSpinner("MyTask")
    spinner.start()
    spinner.stop(success=False)
    err = capsys.readouterr().err
    assert "✗" in err


# ---------------------------------------------------------------------------
# TrainingSpinner — update
# ---------------------------------------------------------------------------

def test_spinner_update_stores_stats(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("Test")
    spinner.start()
    spinner.update({"epoch": "1/5", "loss": "0.1234"})
    with spinner._lock:
        assert spinner._stats["epoch"] == "1/5"
        assert spinner._stats["loss"] == "0.1234"
    spinner.stop()


def test_spinner_update_overwrites_existing_key(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("Test")
    spinner.start()
    spinner.update({"epoch": "1/5"})
    spinner.update({"epoch": "2/5"})
    with spinner._lock:
        assert spinner._stats["epoch"] == "2/5"
    spinner.stop()


def test_spinner_update_merges_multiple_keys(monkeypatch):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("Test")
    spinner.start()
    spinner.update({"a": "1"})
    spinner.update({"b": "2"})
    with spinner._lock:
        assert "a" in spinner._stats
        assert "b" in spinner._stats
    spinner.stop()


# ---------------------------------------------------------------------------
# TrainingSpinner — elapsed time in output
# ---------------------------------------------------------------------------

def test_spinner_output_contains_elapsed_time(monkeypatch, capsys):
    monkeypatch.setattr("utils.progress._ANSI", False)
    spinner = TrainingSpinner("Timed")
    spinner.start()
    spinner.stop(success=True)
    err = capsys.readouterr().err
    # Elapsed time is formatted as MM:SS
    assert ":" in err
