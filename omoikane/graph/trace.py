import numpy as np

from omoikane.graph.autodiff import reverse_mode_mapping


class Node:
    """Wrapper object to trace operations."""

    def __init__(self, ndarray=None, parents=None, op_applied=None):
        self.ndarray = np.array(ndarray) if ndarray is not None else None
        self.parents = parents if parents is not None else []
        self.op_applied = op_applied if op_applied is not None else None
        self.gradients = None

    @classmethod
    def __from_value(cls, value):
        if type(value) == Node:
            return value
        else:
            return Node(value)

    @property
    def T(self):
        return self.__transform(self.ndarray.T)

    @property
    def shape(self):
        return self.ndarray.shape

    @property
    def is_initialized(self):
        return self.ndarray is not None

    def __transform(self, new_value):
        return Node(new_value, self.parents, self.op_applied)

    def reshape(self, new_shape):
        return self.__transform(self.ndarray.reshape(new_shape))

    def __repr__(self):
        return self.ndarray.__str__()

    def __add__(self, other):
        onode = Node.__from_value(other)
        return Node(self.ndarray + onode.ndarray, [self, onode], "add")

    def __radd__(self, other):
        onode = Node.__from_value(other)
        return Node(onode.ndarray + self.ndarray, [onode, self], "add")

    def __sub__(self, other):
        onode = Node.__from_value(other)
        return Node(self.ndarray - onode.ndarray, [self, onode], "sub")

    def __rsub__(self, other):
        onode = Node.__from_value(other)
        return Node(onode.ndarray - self.ndarray, [onode, self], "sub")

    def __mul__(self, other):
        onode = Node.__from_value(other)
        return Node(self.ndarray * onode.ndarray, [self, onode], "mul")

    def __rmul__(self, other):
        onode = Node.__from_value(other)
        return Node(onode.ndarray * self.ndarray, [onode, self], "mul")

    def __truediv__(self, other):
        onode = Node.__from_value(other)
        return Node(self.ndarray / onode.ndarray, [self, onode], "div")

    def __rtruediv__(self, other):
        onode = Node.__from_value(other)
        return Node(onode.ndarray / self.ndarray, [onode, self], "div")

    def __matmul__(self, other):
        onode = Node.__from_value(other)
        return Node(self.ndarray @ onode.ndarray, [self, onode], "matmul")

    def __pow__(self, other):
        onode = Node.__from_value(other)
        return Node(self.ndarray ** onode.ndarray, [self, onode], "pow")

    def __rpow__(self, other):
        onode = Node.__from_value(other)
        return Node(onode.ndarray ** self.ndarray, [onode, self],"pow")

    def __neg__(self):
        return self * Node(-1)

    def backward(self):
        """Performs reverse mode autodiff."""
        if not self.is_initialized:
            raise ValueError("Node not initialized.")
        self.gradients = np.ones(self.ndarray.shape)
        for i, p in enumerate(self.parents):
            p.__backward_recursive(self, self.parents, i)

    def __backward_recursive(self, child, parents, current_idx):
        try:
            # print(self.ndarray, self.gradients, self.parents)
            grad_func = reverse_mode_mapping[child.op_applied]
        except KeyError as e:
            raise NotImplementedError(f"Op '{child.op_applied}' not "
                                      f"implemented in backward mode.")
        if not self.gradients:
            self.gradients = grad_func(child, parents, current_idx)
        else:
            self.gradients += grad_func(child, parents, current_idx)
        # print(self.ndarray, self.gradients, self.parents)
        for i, p in enumerate(self.parents):
            p.__backward_recursive(self, self.parents, i)


def trace(node):
    print(node, node.op_applied)
    for p in node.parents:
        trace(p)
