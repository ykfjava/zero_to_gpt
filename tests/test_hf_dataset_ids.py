import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "data"))

from text_data import OpusBooksDataset, WikiTextDataset  # noqa: E402


class TestHuggingFaceDatasetIds(unittest.TestCase):
    def test_wikitext_uses_namespaced_hub_id(self):
        self.assertEqual(WikiTextDataset.dataset_name, "Salesforce/wikitext")
        self.assertEqual(WikiTextDataset.data_config, "wikitext-103-v1")

    def test_opus_books_uses_namespaced_hub_id(self):
        self.assertEqual(OpusBooksDataset.dataset_name, "Helsinki-NLP/opus_books")
        self.assertEqual(OpusBooksDataset.data_config, "en-es")

    def test_tokenizer_cache_path_has_no_slash(self):
        wrapper = WikiTextDataset(download_split_pct="2%")
        self.assertNotIn("/", wrapper.tokenizer_filename)
        self.assertTrue(wrapper.tokenizer_filename.startswith("Salesforce_wikitext_"))

        opus = OpusBooksDataset(download_split_pct="5%")
        self.assertNotIn("/", opus.tokenizer_filename)
        self.assertTrue(opus.tokenizer_filename.startswith("Helsinki-NLP_opus_books_"))

    def test_load_dataset_passes_namespaced_wikitext_id(self):
        wrapper = WikiTextDataset(download_split_pct="2%")
        with patch("text_data.datasets.load_dataset") as mock_load:
            mock_load.return_value = MagicMock()
            wrapper.load_dataset()
            args, kwargs = mock_load.call_args
            self.assertEqual(args[0], "Salesforce/wikitext")
            self.assertEqual(args[1], "wikitext-103-v1")
            self.assertEqual(kwargs["split"], "train[:2%]")

    def test_load_dataset_passes_namespaced_opus_id(self):
        wrapper = OpusBooksDataset(download_split_pct="5%")
        with patch("text_data.datasets.load_dataset") as mock_load:
            mock_load.return_value = MagicMock()
            wrapper.load_dataset()
            args, kwargs = mock_load.call_args
            self.assertEqual(args[0], "Helsinki-NLP/opus_books")
            self.assertEqual(args[1], "en-es")
            self.assertEqual(kwargs["split"], "train[:5%]")

    def test_unnamespaced_hub_ids_are_rejected_by_datasets(self):
        import datasets

        with self.assertRaises(Exception) as ctx:
            datasets.load_dataset("wikitext", "wikitext-103-v1", split="train[:1]")
        self.assertIn("namespace/name", str(ctx.exception).lower())

        with self.assertRaises(Exception) as ctx:
            datasets.load_dataset("opus_books", "en-es", split="train[:1]")
        self.assertIn("namespace/name", str(ctx.exception).lower())

    def test_namespaced_ids_load_one_row(self):
        import datasets

        wiki = datasets.load_dataset(
            WikiTextDataset.dataset_name,
            WikiTextDataset.data_config,
            split="train[:1]",
        )
        self.assertGreaterEqual(len(wiki), 1)
        self.assertIn("text", wiki.column_names)

        opus = datasets.load_dataset(
            OpusBooksDataset.dataset_name,
            OpusBooksDataset.data_config,
            split="train[:1]",
        )
        self.assertGreaterEqual(len(opus), 1)
        self.assertIn("translation", opus.column_names)
        self.assertIn("en", opus[0]["translation"])
        self.assertIn("es", opus[0]["translation"])


if __name__ == "__main__":
    unittest.main()
