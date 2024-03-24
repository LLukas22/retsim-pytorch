import sys
from unittest.mock import Mock

sys.modules["nmslib"] = Mock()

import numpy as np
from retvec.tf.models.layers import ScaledNorm as TFScaledNorm
from src.retsim_pytorch.modules.norm import ScaledNorm
import torch
import tensorflow as tf
from torch.testing import assert_close
import pytest


@pytest.mark.parametrize("length", [10, 256, 512, 1024])
def test_ScaledNorm(length: int):
    tf_module = TFScaledNorm()
    pt_module = ScaledNorm()

    # Test the forward method
    np.random.seed(0)

    for i in range(10):
        inputs = np.random.rand(1, length, 256)

        tf_output: torch.Tensor = torch.tensor(
            tf_module(tf.constant(inputs, dtype=tf.float32)).numpy()
        )
        pt_output: torch.Tensor = pt_module(torch.tensor(inputs, dtype=torch.float32))

        assert_close(pt_output, tf_output)
