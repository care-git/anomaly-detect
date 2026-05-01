# tests/models/test_svm.py

import unittest
import tempfile
import os
import numpy as np

from models.svm import SVMModel


class TestSVMModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "svm_model")

        # 6 samples (3 per class) so CalibratedClassifierCV(cv=3) has enough folds
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
        self.y = np.array([0, 1, 0, 1, 0, 1])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_and_evaluate_metrics(self):
        model = SVMModel(input_dim=2)
        model.train(self.X, y=self.y)
        metrics = model.evaluate(self.X, self.y)

        self.assertIn("f1_score", metrics)
        self.assertGreaterEqual(metrics["f1_score"], 0)

    def test_predict_returns_expected_labels(self):
        model = SVMModel(input_dim=2)
        model.train(self.X, y=self.y)
        preds = model.predict(self.X)

        self.assertEqual(len(preds), len(self.X))
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_save_and_load_model(self):
        model = SVMModel(input_dim=2)
        model.train(self.X, y=self.y)
        model.save(self.model_path, metrics={"f1_score": 1.0})

        loaded = SVMModel()
        loaded.load(self.model_path)
        preds = loaded.predict(self.X)

        self.assertEqual(len(preds), len(self.X))
        metadata = loaded.get_metadata(self.model_path)
        self.assertIn("model_type", metadata)
        self.assertTrue(metadata["model_type"].lower().startswith("svm"))


if __name__ == "__main__":
    unittest.main()