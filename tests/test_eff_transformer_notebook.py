import json
import unittest
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "eff_transformer" / "eff_transformer.ipynb"


def notebook_sources():
    with NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def source_containing(snippet):
    for source in notebook_sources():
        if snippet in source:
            return source
    raise AssertionError(f"Notebook source containing {snippet!r} was not found")


class EfficientTransformerNotebookTest(unittest.TestCase):
    def test_values_use_separate_kv_projection_weight(self):
        source = source_containing("class MultiHeadAttention")

        self.assertIn(
            'proj_keys = torch.einsum("...se, eo->...so", keys, self.kv_proj_weight[0])',
            source,
        )
        self.assertIn(
            'proj_values = torch.einsum("...se, eo->...so", values, self.kv_proj_weight[1])',
            source,
        )
        self.assertNotIn(
            'proj_values = torch.einsum("...se, eo->...so", values, self.kv_proj_weight[0])',
            source,
        )

    def test_validation_loss_uses_teacher_forced_logits(self):
        source = source_containing("PRINT_VALID = True")

        validation_loss = (
            "pred = model(sequence.to(DEVICE))\n"
            "                loss = loss_fn(pred.reshape(-1, pred.shape[-1]), target.reshape(-1).to(DEVICE))"
        )
        generation = "generated_pred = model(outputs)"

        self.assertIn(validation_loss, source)
        self.assertIn(generation, source)
        self.assertLess(source.index(validation_loss), source.index(generation))
        self.assertIn("sents = generate(sequence, generated_pred, train_wrapper)", source)

    def test_profiler_uses_decoder_only_forward_signature(self):
        source = source_containing("from torch.profiler import profile")

        self.assertIn("model(sequence.to(DEVICE))", source)
        self.assertNotIn("prev_target", source)


if __name__ == "__main__":
    unittest.main()
