# tests/test_preprocessor.py

import unittest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from core.preprocessor import (
    extract_packet_fields,
    clean_dataframe,
    preprocess_file
)


class TestPreprocessor(unittest.TestCase):

    def test_extract_packet_fields_minimal_ip(self):
        mock_pkt = MagicMock()
        mock_pkt.time = 123456.789
        mock_pkt.__len__.return_value = 128
        mock_pkt.haslayer.side_effect = lambda x: x.__name__ == "IP"
        mock_pkt.__contains__.side_effect = lambda x: x.__name__ == "IP"

        # Patch IP extraction
        with patch("core.preprocessor.IP", create=True) as mock_ip:
            mock_ip.__name__ = "IP"
            mock_pkt.__getitem__.return_value.proto = 6
            row = extract_packet_fields(mock_pkt)
            self.assertIsInstance(row, dict)
            self.assertIn("timestamp", row)

    def test_clean_dataframe_handles_missing_values(self):
        df = pd.DataFrame({
            "a": [1, 2, np.nan],
            "b": [np.inf, 2, 3],
            "src": ["192.168.0.1"] * 3,
            "dst": ["192.168.0.2"] * 3
        })
        clean_df = clean_dataframe(df)
        self.assertNotIn("src", clean_df.columns)
        self.assertNotIn("dst", clean_df.columns)
        self.assertFalse(clean_df.isna().any().any())

    @patch("core.preprocessor.PcapReader")
    def test_preprocess_pcap_outputs_dataframe(self, mock_reader):
        mock_pkt = MagicMock()
        mock_pkt.time = 123456.0
        mock_pkt.__len__.return_value = 100
        mock_reader.return_value.__iter__.return_value = [mock_pkt] * 3

        with tempfile.NamedTemporaryFile(suffix=".pcap", delete_on_close=False) as tmp_pcap:
            tmp_pcap.write(b"\x00" * 100)  # simulate binary pcap content
            tmp_pcap.flush()

            with patch("core.preprocessor.config", {"preprocessing": {"batch_size": 2}}):
                df = preprocess_file(tmp_pcap.name, batch_size=2, label=1)

            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn("timestamp", df.columns)


if __name__ == "__main__":
    unittest.main()
