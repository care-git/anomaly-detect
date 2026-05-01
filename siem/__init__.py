# siem/__init__.py

from siem.wazuh_forwarder import forward_alert

__all__ = [
    "forward_alert",
]
