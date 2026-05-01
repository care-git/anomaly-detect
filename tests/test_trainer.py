# tests/test_trainer.py

import unittest
import tempfile
import os
import pandas as pd
from models.trainer import (
    train_autoencoder,
    train_random_forest,
    train_svm
)


class TestTrainerModule(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "model_output")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_autoencoder_on_dummy_data(self):
        df = pd.DataFrame({
            "f1": [0.1, 0.2, 0.3, 0.4],
            "f2": [1.0, 0.9, 1.1, 1.0],
            "f3": [3.5, 3.6, 3.7, 3.5]
        })
        input_path = os.path.join(self.temp_dir.name, "ae_data.csv")
        df.to_csv(input_path, index=False)

        metrics = train_autoencoder(input_path, output_path=self.output_path)
        self.assertIsInstance(metrics, dict)
        self.assertGreater(metrics.get("mse_mean", 0), 0)

    def test_train_random_forest_on_labelled_data(self):
        df = pd.DataFrame({
            "f1": [1, 2, 3, 4],
            "f2": [9, 8, 7, 6],
            "label": [0, 1, 0, 1]
        })
        input_path = os.path.join(self.temp_dir.name, "rf_data.csv")
        df.to_csv(input_path, index=False)

        metrics = train_random_forest(input_path, output_path=self.output_path)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)

    def test_train_svm_on_labelled_data(self):
        # 10 samples (5 per class) so after 80/20 split CalibratedClassifierCV(cv=3)
        # has at least 3 examples per class in the training fold
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "f2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        input_path = os.path.join(self.temp_dir.name, "svm_data.csv")
        df.to_csv(input_path, index=False)

        metrics = train_svm(input_path, output_path=self.output_path)
        self.assertIsInstance(metrics, dict)
        self.assertIn("precision", metrics)


if __name__ == "__main__":
    unittest.main()