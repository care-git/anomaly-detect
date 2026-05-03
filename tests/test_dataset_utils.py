# tests/test_dataset_utils.py

import unittest
import tempfile
import os
import pandas as pd
from core.dataset_utils import (
    load_dataset,
    build_combined_dataset,
    split_dataset,
    balance_labels
)


class TestDatasetUtils(unittest.TestCase):

    def test_load_dataset_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp_file:
            tmp_file.write("a,b,label\n1,2,0\n3,4,1")
            tmp_file.close()
            df = load_dataset(tmp_file.name)
            self.assertEqual(len(df), 2)
            os.remove(tmp_file.name)

    def test_load_dataset_from_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df1 = pd.DataFrame({"a": [1], "b": [2], "label": [0]})
            df2 = pd.DataFrame({"a": [3], "b": [4], "label": [1]})
            df1.to_csv(os.path.join(tmp_dir, "one.csv"), index=False)
            df2.to_csv(os.path.join(tmp_dir, "two.csv"), index=False)

            combined = load_dataset(tmp_dir)
            self.assertEqual(len(combined), 2)

    def test_build_combined_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            df1 = pd.DataFrame({"a": [1], "b": [2], "label": [0]})
            df2 = pd.DataFrame({"a": [1], "b": [2], "label": [0]})  # duplicate - should be dropped
            path1 = os.path.join(tmp_dir, "f1.csv")
            path2 = os.path.join(tmp_dir, "f2.csv")
            df1.to_csv(path1, index=False)
            df2.to_csv(path2, index=False)

            output = os.path.join(tmp_dir, "combined.csv")
            combined = build_combined_dataset([path1, path2], output)
            self.assertEqual(len(combined), 1)  # dedup removes the identical row
            self.assertTrue(os.path.exists(output))

    def test_split_dataset_outputs_three_parts(self):
        df = pd.DataFrame({
            "a": list(range(100)),
            "b": list(range(100, 200)),
            "label": [0, 1] * 50
        })
        train, test = split_dataset(df)
        self.assertAlmostEqual(len(train), 80, delta=2)
        self.assertAlmostEqual(len(test), 20, delta=1)

    def test_balance_labels_equal_output(self):
        df = pd.DataFrame({
            "a": [1]*5 + [2]*10,
            "label": [0]*5 + [1]*10
        })
        balanced = balance_labels(df)
        self.assertEqual(balanced['label'].value_counts()[0], 5)
        self.assertEqual(balanced['label'].value_counts()[1], 5)


if __name__ == "__main__":
    unittest.main()
