import numpy as np

reverse_mode_mapping = {
    "add": (lambda y, X, i: y.gradients),
    "sub": (lambda y, X, i: y.gradients if i == 0 else -y.gradients),
    "mul": (
        lambda y, X, i: y.gradients * X[1].ndarray
        if i == 0
        else y.gradients * X[0].ndarray
    ),
    "log": (lambda y, X, i: y.gradients / X[i].ndarray),
    "sin": (lambda y, X, i: y.gradients * np.cos(X[i].ndarray)),
}
