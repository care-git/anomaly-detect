# tests/test_utils/test_file_saver.py

import unittest
import tempfile
import os
from keras import Sequential
from keras.layers import Dense
from utils.file_saver import (
    save_pickle,
    save_json,
    save_keras_model
)


class TestFileSaverUtils(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = os.path.join(self.temp_dir.name, "output")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_pickle_creates_file(self):
        data = {"key": "value", "arr": [1, 2, 3]}
        path = os.path.join(self.temp_dir.name, "test.pkl")
        save_pickle(data, path)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.getsize(path) > 0)

    def test_save_json_creates_file(self):
        metadata = {"model_type": "test", "accuracy": 0.95}
        path = os.path.join(self.temp_dir.name, "meta.json")
        save_json(metadata, path)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.getsize(path) > 0)

    def test_save_keras_model_creates_file(self):
        model = Sequential([Dense(4, input_shape=(3,), activation='relu'), Dense(2)])
        path = os.path.join(self.temp_dir.name, "keras_model.keras")
        save_keras_model(model, path)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.getsize(path) > 0)


if __name__ == "__main__":
    unittest.main()
