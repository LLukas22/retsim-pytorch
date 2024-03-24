import sys
from unittest.mock import Mock

sys.modules["nmslib"] = Mock()

from src.retsim_pytorch.modules import GeneralizedMeanPooling1D
import torch
import numpy as np
import pytest
import math
from torch.testing import assert_close


def test_is_constructable():
    pt_module = GeneralizedMeanPooling1D(p=3.0)

    assert pt_module is not None


@pytest.mark.parametrize("dim", [1, 64, 128, 256, 512])
def test_output_sizes_match(dim: int):
    pt_module = GeneralizedMeanPooling1D(p=3.0)
    inputs = torch.rand(1, dim, 256)

    pt_output = pt_module(inputs)

    assert pt_output.shape == (1, 256)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, math.inf, -math.inf, 0.0000000001])
def test_results_match(p: float):
    from tensorflow_similarity.layers import (
        GeneralizedMeanPooling1D as TFGeneralizedMeanPooling1D,
    )
    import tensorflow as tf

    tf_module = TFGeneralizedMeanPooling1D(p=p)
    pt_module = GeneralizedMeanPooling1D(p=p)

    inputs = np.random.rand(1, 512, 256)

    tf_result = torch.tensor(tf_module(tf.constant(inputs, dtype=tf.float32)).numpy())
    pt_module = pt_module.eval()
    pt_result = pt_module(torch.tensor(inputs, dtype=torch.float32))

    assert_close(tf_result, pt_result)
