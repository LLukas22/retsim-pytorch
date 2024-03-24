import sys
from unittest.mock import Mock

sys.modules["nmslib"] = Mock()

import numpy as np
import pytest
from retvec.tf.models.layers import get_activation_layer as tf_get_activation_layer
from src.retsim_pytorch.modules import get_activation_layer
import torch
import tensorflow as tf
from torch.testing import assert_close


@pytest.mark.parametrize("activation", ["relu1", "relu2", "sqrrelu"])
def test_activations(activation: str):
    tf_layer = tf_get_activation_layer(activation)
    pt_layer = get_activation_layer(activation)

    input = np.arange(-10, 10, 0.1)
    tf_output = torch.tensor(tf_layer(tf.constant(input, dtype=tf.float32)).numpy())
    pt_output = pt_layer(torch.tensor(input, dtype=torch.float32))

    assert_close(pt_output, tf_output)
