import os
import sys
import unittest

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
from padding import PaddingSampler, pad_collate


class PaddingSamplerTests(unittest.TestCase):
    def test_accepts_python_list(self):
        sampler = PaddingSampler([3, 1, 2, 5])
        self.assertEqual(sorted(sampler), [0, 1, 2, 3])

    def test_accepts_torch_tensor(self):
        sampler = PaddingSampler(torch.tensor([3, 1, 2, 5]))
        self.assertEqual(sorted(sampler), [0, 1, 2, 3])

    def test_accepts_huggingface_column(self):
        from datasets import Dataset

        ds = Dataset.from_dict({"en_lens": [3, 1, 2, 5]})
        sampler = PaddingSampler(ds["en_lens"])
        self.assertEqual(sorted(sampler), [0, 1, 2, 3])

        torch_ds = ds.with_format("torch")
        sampler = PaddingSampler(torch_ds["en_lens"])
        self.assertEqual(sorted(sampler), [0, 1, 2, 3])

    def test_yields_length_grouped_indices(self):
        lengths = [10, 2, 10, 2, 10, 2]
        order = list(PaddingSampler(lengths))
        self.assertEqual(sorted(order), list(range(len(lengths))))
        sampled_lengths = [lengths[i] for i in order]
        self.assertEqual(sampled_lengths, sorted(sampled_lengths))

    def test_dataloader_accepts_huggingface_style_column(self):
        from datasets import Dataset

        lengths = [4, 2, 4, 2]
        ds = Dataset.from_dict({
            "en_ids": [list(range(1, n + 1)) for n in lengths],
            "es_ids": [list(range(1, n + 1)) for n in lengths],
            "en_lens": lengths,
        }).with_format("torch")
        loader = DataLoader(
            ds,
            batch_size=2,
            sampler=PaddingSampler(ds["en_lens"]),
            collate_fn=pad_collate,
        )
        rows = sum(len(batch["en_ids"]) for batch in loader)
        self.assertEqual(rows, len(lengths))


if __name__ == "__main__":
    unittest.main()
