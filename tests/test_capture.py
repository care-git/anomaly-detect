# tests/test_capture.py

import unittest
import tempfile
import os
from unittest.mock import patch
from core.capture import capture_packets, live_packet_monitor


class TestCaptureModule(unittest.TestCase):

    @patch("core.capture.save_pcap")
    @patch("core.capture.sniff")
    def test_capture_packets_calls_sniff(self, mock_sniff, mock_save):
        mock_sniff.side_effect = lambda **kwargs: kwargs['prn']("mock_packet")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "capture.pcap")
            capture_packets(interface="lo", duration=1, packet_count=5, output_path=output_path)

        self.assertTrue(mock_sniff.called)

    @patch("core.capture.sniff")
    def test_live_packet_monitor_triggers_callback(self, mock_sniff):
        packets = []

        def fake_sniff(**kwargs):
            # Simulate callback for 3 packets
            for _ in range(3):
                kwargs['prn']("packet")
            return ["packet"] * 3

        mock_sniff.side_effect = fake_sniff

        # Test callback receives packets
        received = []
        def callback(pkt):
            received.append(pkt)

        live_packet_monitor(interface="lo", packet_callback=callback, count=3, timeout=1)

        self.assertEqual(len(received), 3)
        self.assertEqual(received, ["packet", "packet", "packet"])


if __name__ == "__main__":
    unittest.main()
