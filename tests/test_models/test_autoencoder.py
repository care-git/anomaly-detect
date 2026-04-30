# tests/models/test_autoencoder.py

import unittest
import tempfile
import os
import numpy as np

from models.autoencoder import AutoencoderModel


class TestAutoencoderModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "ae_model")

        # Synthetic unsupervised dataset (4 samples, 3 features)
        self.X = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.3, 0.2, 0.1],
            [0.6, 0.5, 0.4]
        ])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_model_constructs_without_error(self):
        model = AutoencoderModel(input_dim=3)
        net = model.build_model()
        self.assertEqual(net.input_shape[1], 3)

    def test_train_and_evaluate_outputs_metrics(self):
        model = AutoencoderModel(input_dim=3)
        model.train(self.X, X_val=self.X)
        metrics = model.evaluate(self.X)

        self.assertIn("mse_mean", metrics)
        self.assertGreater(metrics["mse_mean"], 0)

    def test_model_save_and_reload_works(self):
        model = AutoencoderModel(input_dim=3)
        model.train(self.X, X_val=self.X)
        model.save(self.output_path, metrics={"mse_mean": 0.1})

        new_model = AutoencoderModel()
        new_model.load(self.output_path)

        self.assertIsNotNone(new_model.model)
        self.assertIsNotNone(new_model.scaler)
        self.assertGreater(new_model.threshold, 0)
        metadata = new_model.get_metadata(self.output_path)
        self.assertIn("model_type", metadata)


if __name__ == "__main__":
    unittest.main()