# tests/test_utils/test_progress.py

import unittest
from utils.progress import tqdm_bar


class TestProgressUtils(unittest.TestCase):

    def test_tqdm_bar_outputs_correct_length(self):
        data = list(range(5))
        output = []

        for i in tqdm_bar(data, desc="Testing tqdm_bar", unit="it", leave=False):
            output.append(i)

        self.assertEqual(output, data)


if __name__ == "__main__":
    unittest.main()
