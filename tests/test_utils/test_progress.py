# tests/test_utils/test_progress.py

import unittest
from utils.progress import tqdm_bar, single_bar


class TestProgressUtils(unittest.TestCase):

    def test_tqdm_bar_outputs_correct_length(self):
        data = list(range(5))
        output = []

        for i in tqdm_bar(data, desc="Testing tqdm_bar", unit="it", leave=False):
            output.append(i)

        self.assertEqual(output, data)

    def test_single_bar_context_manager_completes(self):
        called = []
        with single_bar("Testing single_bar", total=1, unit="step", leave=False) as update:
            called.append(True)
            update()
        self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()
