import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
from csv_data import WeatherDatasetWrapper, split_dataframe


class SplitDataframeTest(unittest.TestCase):
    def test_split_keeps_dataframes_and_column_access(self):
        data = pd.DataFrame({
            "tmax": np.arange(100, dtype=float),
            "tmin": np.arange(100, dtype=float) - 10,
            "rain": np.zeros(100),
            "tmax_tomorrow": np.arange(100, dtype=float) + 1,
        })
        parts = split_dataframe(data)
        self.assertEqual(len(parts), 3)
        self.assertEqual([len(p) for p in parts], [70, 15, 15])
        for part in parts:
            self.assertIsInstance(part, pd.DataFrame)
            values = part[["tmax", "tmin", "rain"]].to_numpy()
            self.assertEqual(values.shape[1], 3)

    def test_splits_cover_every_row_in_order(self):
        data = pd.DataFrame({"x": np.arange(10), "y": np.arange(10)})
        parts = split_dataframe(data)
        restored = pd.concat(parts)
        pd.testing.assert_frame_equal(restored.reset_index(drop=True), data.reset_index(drop=True))

    def test_legacy_np_split_breaks_column_indexing_on_pandas3(self):
        if pd.__version__.split(".", 1)[0] < "3":
            self.skipTest("np.split DataFrame regression is pandas 3+")
        data = pd.DataFrame({
            "tmax": np.arange(20, dtype=float),
            "tmin": np.arange(20, dtype=float),
            "rain": np.zeros(20),
            "tmax_tomorrow": np.arange(20, dtype=float),
        })
        split_data = np.split(data, [int(.7 * len(data)), int(.85 * len(data))])
        with self.assertRaises(IndexError):
            _ = [d[["tmax", "tmin", "rain"]].to_numpy() for d in split_data]


class WeatherSplitTest(unittest.TestCase):
    def test_weather_wrapper_builds_numpy_splits(self):
        wrapper = WeatherDatasetWrapper()
        total = 0
        for split_name in ["train", "validation", "test"]:
            x = wrapper.split_data[split_name]["x"]
            y = wrapper.split_data[split_name]["target"]
            self.assertIsInstance(x, np.ndarray)
            self.assertIsInstance(y, np.ndarray)
            self.assertEqual(x.shape[0], y.shape[0])
            self.assertEqual(x.shape[1], 3)
            total += x.shape[0]
        self.assertEqual(total, len(wrapper.data))


if __name__ == "__main__":
    unittest.main()
