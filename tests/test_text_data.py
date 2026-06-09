import sys
import types
import unittest


def _install_dependency_stubs():
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.SentencePieceBPETokenizer = object
    sys.modules.setdefault("tokenizers", tokenizers)

    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizerFast = object
    sys.modules.setdefault("transformers", transformers)

    datasets = types.ModuleType("datasets")
    sys.modules.setdefault("datasets", datasets)


_install_dependency_stubs()

from data.text_data import OpusBooksDataset, WikiTextDataset


class FakeDataset:
    def __init__(self, data_key, values):
        self.data_key = data_key
        self.values = values
        self.column_names = [data_key]
        self.map_kwargs = None

    def map(self, func, **kwargs):
        self.map_kwargs = kwargs
        return func({self.data_key: self.values})


class TextDataTests(unittest.TestCase):
    def test_wikitext_combines_whole_split_to_preserve_cross_batch_articles(self):
        wrapper = WikiTextDataset(processes=4)
        article_prefix = "a" * 999
        article_suffix = "continued"
        data = FakeDataset(
            wrapper.data_key,
            [
                " = Article One = \n",
                article_prefix,
                article_suffix,
                " = Article Two = \n",
                "second",
            ],
        )

        combined = wrapper.combine_dataset(data)

        self.assertIsNone(data.map_kwargs["batch_size"])
        self.assertIsNone(data.map_kwargs["num_proc"])
        self.assertEqual(
            combined,
            {
                wrapper.data_key: [
                    article_prefix + article_suffix,
                    "second",
                ]
            },
        )

    def test_default_combine_dataset_keeps_existing_batched_map_settings(self):
        wrapper = OpusBooksDataset(processes=3)
        examples = [
            {"en": "hello", "es": "hola"},
            {"en": "goodbye", "es": "adios"},
        ]
        data = FakeDataset(wrapper.data_key, examples)

        combined = wrapper.combine_dataset(data)

        self.assertNotIn("batch_size", data.map_kwargs)
        self.assertEqual(data.map_kwargs["num_proc"], 3)
        self.assertEqual(
            combined,
            {
                wrapper.data_key: [
                    "hello<s>hola",
                    "goodbye<s>adios",
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
