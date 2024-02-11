import numpy as np

import omoikane as ok
from omoikane import Node


def test_autodiff_reverse_mode_should_return_correct_gradients():
    x_1 = Node(2)
    x_2 = Node(5)
    v_1 = ok.log(x_1)
    v_2 = x_1 * x_2
    v_3 = ok.sin(x_2)
    v_4 = v_1 + v_2
    v_5 = v_4 - v_3
    v_5.backward()
    assert v_5.gradients == 1
    assert v_4.gradients == 1
    assert v_3.gradients == -1
    assert v_2.gradients == 1
    assert v_1.gradients == 1
    assert np.isclose(x_2.gradients, 1.7163378145367738)
    assert np.isclose(x_1.gradients, 5.5)
    # print(v_4.gradients
