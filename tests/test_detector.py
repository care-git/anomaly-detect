# tests/test_detector.py

import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import patch

from models.detector import run_detection
from models.trainer import train_random_forest


class TestDetectorModule(unittest.TestCase):

    def setUp(self):
        # Setup temp dirs and data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_dir = os.path.join(self.temp_dir.name, "rf_model")
        self.input_csv = os.path.join(self.temp_dir.name, "input.csv")
        self.output_csv = os.path.join(self.temp_dir.name, "output.csv")

        # Generate and train a dummy RF model
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "f2": [9, 8, 7, 6],
            "label": [0, 1, 0, 1]
        })
        df.to_csv(self.input_csv, index=False)
        train_random_forest(self.input_csv, output_path=self.model_dir)

        # Create test CSV without the label (mimic preprocessed)
        test_df = df.drop(columns=["label"])
        test_df.to_csv(self.input_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("models.detector.forward_alert")
    def test_run_detection_generates_predictions_and_alerts(self, mock_alert):
        class Args:
            model = "random_forest"
            model_path = self.model_dir
            input = self.input_csv
            output = self.output_csv
            live = False
            interface = None

        run_detection(Args())

        self.assertTrue(os.path.exists(self.output_csv))
        df = pd.read_csv(self.output_csv)
        self.assertIn("prediction", df.columns)
        self.assertTrue(mock_alert.called)

    def test_detector_handles_empty_input_gracefully(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("")  # Empty CSV
            tmp.close()

            class Args:
                model = "random_forest"
                model_path = self.model_dir
                input = tmp.name
                output = self.output_csv
                live = False
                interface = None

            with patch("models.detector.logger") as mock_logger:
                run_detection(Args())
                self.assertTrue(mock_logger.error.called)

            os.remove(tmp.name)


if __name__ == "__main__":
    unittest.main()