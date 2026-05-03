# tests/test_siem/test_wazuh_forwarder.py

import json
import os
from unittest.mock import patch

import pytest

import siem.wazuh_forwarder as wf
from siem.wazuh_forwarder import forward_alert


def _siem_cfg(mode, log_path):
    return {"siem": {
        "mode": mode,
        "log_path": log_path,
        "log_max_bytes": 1024 * 1024,
        "log_backup_count": 2,
        "syslog_host": "127.0.0.1",
        "syslog_port": 514,
    }}


@pytest.fixture(autouse=True)
def reset_alert_logger():
    """Reset the module-level _alert_logger singleton between tests."""
    wf._alert_logger = None
    yield
    wf._alert_logger = None


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

def test_file_mode_writes_json_line(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))

    result = forward_alert({"rule": "port_scan", "src_ip": "192.168.1.1"})

    assert result is True
    assert os.path.exists(log_path)
    with open(log_path) as f:
        parsed = json.loads(f.read().strip())
    assert parsed["rule"] == "port_scan"
    assert parsed["src_ip"] == "192.168.1.1"


def test_file_mode_creates_parent_directory(tmp_path, monkeypatch):
    log_path = str(tmp_path / "deep" / "nested" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))
    forward_alert({"rule": "dir_test"})
    assert os.path.exists(log_path)


def test_file_mode_appends_multiple_alerts(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))

    forward_alert({"rule": "first"})
    forward_alert({"rule": "second"})

    with open(log_path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["rule"] == "first"
    assert json.loads(lines[1])["rule"] == "second"


# ---------------------------------------------------------------------------
# Syslog mode
# ---------------------------------------------------------------------------

def test_syslog_mode_sends_udp_packet(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("syslog", log_path))

    sent = []

    class _MockSocket:
        def sendto(self, data, addr):
            sent.append((data, addr))
        def close(self):
            pass

    with patch("siem.wazuh_forwarder.socket.socket", return_value=_MockSocket()):
        result = forward_alert({"rule": "scan"})

    assert result is True
    assert len(sent) == 1
    payload, addr = sent[0]
    assert addr == ("127.0.0.1", 514)
    assert b"scan" in payload


def test_syslog_mode_does_not_write_file(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("syslog", log_path))

    with patch("siem.wazuh_forwarder.socket.socket", return_value=type("S", (), {"sendto": lambda *a: None, "close": lambda *a: None})() ):
        forward_alert({"rule": "no_file"})

    assert not os.path.exists(log_path)


# ---------------------------------------------------------------------------
# Both mode
# ---------------------------------------------------------------------------

def test_both_mode_writes_file_and_sends_udp(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("both", log_path))

    sent = []

    class _MockSocket:
        def sendto(self, data, addr):
            sent.append(data)
        def close(self):
            pass

    with patch("siem.wazuh_forwarder.socket.socket", return_value=_MockSocket()):
        result = forward_alert({"rule": "both_test"})

    assert result is True
    assert os.path.exists(log_path)
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_dry_run_returns_true_without_writing(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))

    result = forward_alert({"rule": "dry"}, dry_run=True)

    assert result is True
    assert not os.path.exists(log_path)


def test_dry_run_syslog_does_not_send(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("syslog", log_path))

    with patch("siem.wazuh_forwarder.socket.socket") as mock_sock_cls:
        forward_alert({"rule": "dry_syslog"}, dry_run=True)
    mock_sock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

def test_timestamp_injected_when_absent(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))

    forward_alert({"rule": "ts_missing"})

    with open(log_path) as f:
        parsed = json.loads(f.read().strip())
    assert "timestamp" in parsed


def test_existing_timestamp_preserved(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))

    fixed_ts = "2026-01-01T00:00:00+00:00"
    forward_alert({"rule": "ts_fixed", "timestamp": fixed_ts})

    with open(log_path) as f:
        parsed = json.loads(f.read().strip())
    assert parsed["timestamp"] == fixed_ts


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------

def test_returns_false_when_all_delivery_paths_fail(tmp_path, monkeypatch):
    log_path = str(tmp_path / "alerts" / "alerts.log")
    monkeypatch.setattr("siem.wazuh_forwarder.get_config", lambda: _siem_cfg("file", log_path))

    with patch("siem.wazuh_forwarder._get_alert_file_logger", side_effect=OSError("disk full")):
        result = forward_alert({"rule": "fail"})

    assert result is False
