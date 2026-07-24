import json
import pathlib
import unittest

import torch
import torch.nn as nn


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "eff_transformer" / "eff_transformer.ipynb"


def notebook_source():
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


class TiedEmbedding(nn.Module):
    """Mirrors Transformer.embed from the efficient transformer notebook."""

    def __init__(self, vocab_size, hidden_units):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, hidden_units))
        nn.init.xavier_uniform_(self.embedding)

    def embed(self, x, reverse=False):
        if reverse:
            return x @ self.embedding.T
        embedded = self.embedding[x.to(torch.long).reshape(-1)]
        return embedded.view(x.shape[0], x.shape[1], -1)

    def forward(self, x):
        return self.embed(x)


class EffTransformerEmbedContiguousTests(unittest.TestCase):
    def test_notebook_embed_uses_reshape_for_token_ids(self):
        source = notebook_source()
        self.assertIn("x.to(torch.long).reshape(-1)", source)
        self.assertNotIn("x.to(torch.long).view(-1)", source)

    def test_cpu_validation_prefix_slice_does_not_crash_embed(self):
        model = TiedEmbedding(vocab_size=64, hidden_units=16)
        sequence = torch.randint(0, 64, (4, 128))
        outputs = sequence[:, :50]
        self.assertFalse(outputs.is_contiguous())

        pred = model(outputs.to("cpu"))
        self.assertEqual(pred.shape, (4, 50, 16))


if __name__ == "__main__":
    unittest.main()
