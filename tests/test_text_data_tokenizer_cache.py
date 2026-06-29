import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(DATA_DIR))


class _SentencePieceBPETokenizer:
    pass


class _PreTrainedTokenizerFast:
    pass


sys.modules.setdefault(
    "tokenizers",
    types.SimpleNamespace(SentencePieceBPETokenizer=_SentencePieceBPETokenizer),
)
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(PreTrainedTokenizerFast=_PreTrainedTokenizerFast),
)
sys.modules.setdefault("datasets", types.SimpleNamespace())

from text_data import OpusBooksDataset, WikiTextDataset


class TokenizerCacheFilenameTest(unittest.TestCase):
    def test_vocab_size_changes_cache_filename(self):
        large_vocab = OpusBooksDataset(tokenizer_vocab=5000, download_split_pct="5%")
        small_vocab = OpusBooksDataset(tokenizer_vocab=1000, download_split_pct="5%")

        self.assertNotEqual(large_vocab.tokenizer_filename, small_vocab.tokenizer_filename)
        self.assertIn("vocab5000", large_vocab.tokenizer_filename)
        self.assertIn("vocab1000", small_vocab.tokenizer_filename)

    def test_training_parameters_and_split_are_part_of_cache_filename(self):
        default_freq = WikiTextDataset(
            tokenizer_vocab=5000,
            min_token_freq=2,
            download_split="train",
            download_split_pct="2%",
        )
        high_freq = WikiTextDataset(
            tokenizer_vocab=5000,
            min_token_freq=10,
            download_split="train",
            download_split_pct="2%",
        )
        validation_split = WikiTextDataset(
            tokenizer_vocab=5000,
            min_token_freq=2,
            download_split="validation",
            download_split_pct="2%",
        )

        self.assertNotEqual(default_freq.tokenizer_filename, high_freq.tokenizer_filename)
        self.assertNotEqual(default_freq.tokenizer_filename, validation_split.tokenizer_filename)
        self.assertIn("minfreq2", default_freq.tokenizer_filename)
        self.assertIn("minfreq10", high_freq.tokenizer_filename)
        self.assertIn("validation", validation_split.tokenizer_filename)

    def test_cache_filename_is_path_safe(self):
        wrapper = OpusBooksDataset(tokenizer_vocab=5000, download_split_pct="5%")

        self.assertNotIn("%", wrapper.tokenizer_filename)
        self.assertNotIn("/", wrapper.tokenizer_filename)
        self.assertTrue(wrapper.tokenizer_filename.endswith("_tokenizer"))


if __name__ == "__main__":
    unittest.main()
