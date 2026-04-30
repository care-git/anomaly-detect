# siem/wazuh_forwarder.py

import json
import logging
import logging.handlers
import os
import socket
from datetime import datetime, timezone
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__, "INFO")

_alert_logger = None


def _get_alert_file_logger(log_path: str, max_bytes: int, backup_count: int) -> logging.Logger:
    """
    Returns (creating if needed) a dedicated rotating-file logger for alert JSON lines.

    Using a separate logger with its own RotatingFileHandler keeps alert output
    isolated from the application log stream and bounds file growth automatically.
    """
    global _alert_logger
    if _alert_logger is None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _alert_logger = logging.getLogger("anomaly_detect.alerts")
        _alert_logger.setLevel(logging.INFO)
        _alert_logger.addHandler(handler)
        _alert_logger.propagate = False
    return _alert_logger


def forward_alert(alert: dict, dry_run: bool = False) -> bool:
    """
    Forwards an alert to Wazuh via file logging or syslog.

    Delivery mode is controlled by `siem.mode` in the config file, which can be:
        - 'file': log alerts to a rotating file
        - 'syslog': send alerts to a syslog server
        - 'both': do both

    Parameters:
        alert (dict): Dictionary containing the alert data.
        dry_run (bool): If True, simulate sending without actually writing/transmitting.

    Returns:
        bool: True if at least one delivery was successful, False otherwise.
    """
    siem_config = get_config()['siem']
    mode = siem_config['mode']
    log_path = siem_config['log_path']
    syslog_addr = siem_config['syslog_host']
    syslog_port = siem_config['syslog_port']
    max_bytes = siem_config.get('log_max_bytes', 10 * 1024 * 1024)   # 10 MB default
    backup_count = siem_config.get('log_backup_count', 5)

    alert = {**alert, "timestamp": alert.get('timestamp') or datetime.now(timezone.utc).isoformat()}
    alert_json = json.dumps(alert)

    success = False

    if mode in ('file', 'both'):
        try:
            logger.debug("Writing alert to file: %s", log_path)
            if not dry_run:
                file_logger = _get_alert_file_logger(log_path, max_bytes, backup_count)
                file_logger.info(alert_json)
            success = True
        except Exception as e:
            logger.error("Failed to write alert to log file: %s", e)

    if mode in ('syslog', 'both'):
        try:
            logger.debug("Sending alert via syslog to %s:%d", syslog_addr, syslog_port)
            if not dry_run:
                # RFC 5424 syslog header: <priority>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID
                priority = 14  # facility=1 (user), severity=6 (info)
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                hostname = socket.gethostname()
                rfc5424_msg = (
                    f"<{priority}>1 {timestamp} {hostname} anomaly-detect - - - {alert_json}"
                ).encode()
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(rfc5424_msg, (syslog_addr, syslog_port))
                sock.close()
            success = True
        except Exception as e:
            logger.error("Failed to send alert via syslog: %s", e)

    return success
