# siem/wazuh_forwarder.py

import json
import os
import socket
from datetime import datetime, timezone
from utils.config_loader import get_config
from utils.logger import get_logger

config = get_config()
logger = get_logger(__name__, config.get("general", {}).get("logging_level", "INFO"))


def forward_alert(alert: dict, dry_run: bool = False) -> bool:
    """
    Forwards an alert to Wazuh via file logging or syslog.

    Delivery mode is controlled by `siem.mode` in the config file, which can be:
        - 'file': log alerts to a file
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

    alert = {**alert, "timestamp": alert.get('timestamp') or datetime.utcnow().isoformat()}
    alert_json = json.dumps(alert)

    success = False

    if mode in ('file', 'both'):
        try:
            logger.debug("Writing alert to file: %s", log_path)
            if not dry_run:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, 'a') as f:
                    f.write(alert_json + '\n')
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
