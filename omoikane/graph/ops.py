import numpy as np

from omoikane.graph.trace import Node


def exp(x: Node):
    return Node(np.exp(x.ndarray), parents=[x], op_applied="exp")


def log(x: Node):
    return Node(np.log(x.ndarray), parents=[x], op_applied="log")


def sin(x: Node):
    return Node(np.sin(x.ndarray), parents=[x], op_applied="sin")


def cos(x: Node):
    return Node(np.cos(x.ndarray), parents=[x], op_applied="cos")


def sigmoid(x: Node):
    return 1 / (1 + exp(-x))


def tanh(x: Node):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
