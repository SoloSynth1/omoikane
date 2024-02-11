import numpy as np

from omoikane import Node


def test_node_should_work():
    x = Node(1.0)
    assert np.array_equal(x.ndarray, np.array(1.0))


def test_arithmetics_should_work():
    vals = [[1.0, 4.53, 193]]
    x = Node(vals)
    x = ((x + x * x) / x) - x
    y = np.array(vals)
    y = ((y + y * y) / y) - y
    assert np.array_equal(x.ndarray, y)
