import sys
import types
import unittest


try:
    import tokenizers  # noqa: F401
except ImportError:
    tokenizers = types.ModuleType("tokenizers")

    class SentencePieceBPETokenizer:
        pass

    tokenizers.SentencePieceBPETokenizer = SentencePieceBPETokenizer
    sys.modules["tokenizers"] = tokenizers

try:
    import transformers  # noqa: F401
except ImportError:
    transformers = types.ModuleType("transformers")

    class PreTrainedTokenizerFast:
        pass

    transformers.PreTrainedTokenizerFast = PreTrainedTokenizerFast
    sys.modules["transformers"] = transformers

try:
    import datasets  # noqa: F401
except ImportError:
    sys.modules["datasets"] = types.ModuleType("datasets")

from data.text_data import WikiTextDataset


class FakeDataset:
    column_names = ["text"]

    def __init__(self, rows):
        self.rows = rows
        self.map_kwargs = None

    def map(self, func, **kwargs):
        self.map_kwargs = kwargs
        return func({"text": self.rows})


class WikiTextDatasetTests(unittest.TestCase):
    def test_chunked_combine_splits_article_at_map_boundaries(self):
        wrapper = WikiTextDataset()
        rows = [
            " = First Article = \n",
            "first half ",
            "second half ",
            " = Second Article = \n",
            "another article ",
        ]

        chunked_entries = []
        for chunk in (rows[:2], rows[2:]):
            chunked_entries.extend(wrapper.combine_func({"text": chunk})["text"])

        self.assertEqual(chunked_entries, ["first half ", "second half ", "another article "])

    def test_wikitext_combines_full_dataset_in_one_map_batch(self):
        wrapper = WikiTextDataset(processes=4)
        rows = [
            " = First Article = \n",
            "first half ",
            "second half ",
            " = Second Article = \n",
            "another article ",
        ]
        dataset = FakeDataset(rows)

        combined = wrapper.combine_dataset(dataset)

        self.assertEqual(combined, {"text": ["first half second half ", "another article "]})
        self.assertTrue(dataset.map_kwargs["batched"])
        self.assertIsNone(dataset.map_kwargs["batch_size"])
        self.assertIsNone(dataset.map_kwargs["num_proc"])


if __name__ == "__main__":
    unittest.main()
