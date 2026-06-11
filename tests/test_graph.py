import pathlib
import sys
import types
import unittest

import numpy as np


graphviz = types.ModuleType("graphviz")


class DummyDigraph:
    def __init__(self, *args, **kwargs):
        pass

    def attr(self, *args, **kwargs):
        pass

    def node(self, *args, **kwargs):
        pass

    def edge(self, *args, **kwargs):
        pass


graphviz.Digraph = DummyDigraph
sys.modules.setdefault("graphviz", graphviz)

ipython = types.ModuleType("IPython")
display = types.ModuleType("IPython.display")
display.Latex = str
sys.modules.setdefault("IPython", ipython)
sys.modules.setdefault("IPython.display", display)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "nnets"))

from graph import Node, Parameter


class Add(Node):
    def forward(self, x, y):
        return x + y

    def backward(self, grad):
        return grad, grad


class Multiply(Node):
    def forward(self, x, y):
        return x * y

    def backward(self, grad):
        x, y = self.cache
        return grad * y, grad * x


class Sum(Node):
    def forward(self, x):
        return np.sum(x).reshape(1, 1)

    def backward(self, grad):
        return np.ones(self.cache[0].shape) * grad


class GraphBackwardTests(unittest.TestCase):
    def test_shared_leaf_gradient_is_applied_once(self):
        x = Parameter(np.array([[3.0]]), desc="x")
        doubled = Add(x, x)
        output = Multiply(doubled, x)

        np.testing.assert_allclose(output.apply_fwd(), np.array([[18.0]]))

        output.zero_grad()
        output.apply_bwd(np.array([[1.0]]))

        np.testing.assert_allclose(x.grad, np.array([[12.0]]))

    def test_shared_broadcasted_subgraph_accumulates_duplicate_edges(self):
        x = Parameter(np.array([[1.0, 2.0], [3.0, 4.0]]), needs_grad=False)
        b = Parameter(np.array([[10.0, 20.0]]), desc="b")
        shifted = Add(x, b)
        doubled = Add(shifted, shifted)
        output = Sum(doubled)

        np.testing.assert_allclose(output.apply_fwd(), np.array([[80.0]]))

        output.zero_grad()
        output.apply_bwd(np.array([[1.0]]))

        np.testing.assert_allclose(b.grad, np.array([[4.0, 4.0]]))


if __name__ == "__main__":
    unittest.main()
