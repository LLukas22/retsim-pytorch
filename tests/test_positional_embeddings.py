import sys
from unittest.mock import Mock

sys.modules["nmslib"] = Mock()

from retvec.tf.models.positional_embeddings import (
    ScaledSinusoidalPositionalEmbedding as TFScaledSinusoidalPositionalEmbedding,
)
from retvec.tf.models.positional_embeddings import rope as tf_rope
from retvec.tf.models.positional_embeddings import toeplitz_matrix as tf_toeplitz_matrix
from retvec.tf.models.positional_embeddings import (
    toeplitz_matrix_rope as tf_toeplitz_matrix_rope,
)
from src.retsim_pytorch.modules import ScaledSinusoidalPositionalEmbedding
from src.retsim_pytorch.modules.positional_embeddings import (
    rope,
    toeplitz_matrix,
    toeplitz_matrix_rope,
)
import numpy as np
import torch
import tensorflow as tf
import pytest
from torch.testing import assert_close


RANDOM_SAMPLES = 10


@pytest.mark.parametrize("hidden_size", [256, 512, 1024])
def test_ScaledSinusoidalPositionalEmbedding_sequence_lengths(hidden_size: int):
    tf_module = TFScaledSinusoidalPositionalEmbedding(hidden_size=24)
    pt_module = ScaledSinusoidalPositionalEmbedding(hidden_size=24)

    # Test the forward method
    np.random.seed(0)

    for i in range(RANDOM_SAMPLES):
        inputs = np.random.rand(1, hidden_size, 24)

        tf_output: torch.Tensor = torch.tensor(
            tf_module(tf.constant(inputs, dtype=tf.float32)).numpy()
        )
        pt_output: torch.Tensor = pt_module(torch.tensor(inputs, dtype=torch.float32))

        assert_close(pt_output, tf_output)


@pytest.mark.parametrize(
    "dim,axis",
    [([1, 10, 256], 1), ([1, 512], 0), ([1, 512, 256], 1), ([1, 1024, 512], 1)],
)
def test_rope(dim: tuple, axis: int):
    np.random.seed(0)

    for i in range(RANDOM_SAMPLES):
        inputs = np.random.rand(*dim)

        tf_output: torch.Tensor = torch.tensor(
            tf_rope(tf.constant(inputs, dtype=tf.float32), axis=axis).numpy()
        )
        pt_output: torch.Tensor = rope(
            torch.tensor(inputs, dtype=torch.float32), axis=axis
        )
        assert_close(pt_output, tf_output, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("max_length", [128, 256, 512, 1024])
def test_toeplitz_matrix(max_length: int):
    np.random.seed(0)

    for i in range(RANDOM_SAMPLES):
        n = np.random.rand(2 * max_length - 1)

        tf_output: torch.Tensor = torch.tensor(
            tf_toeplitz_matrix(max_length, tf.constant(n, dtype=tf.float32)).numpy()
        )
        pt_output: torch.Tensor = toeplitz_matrix(
            max_length, torch.tensor(n, dtype=torch.float32)
        )

        assert_close(pt_output, tf_output)


@pytest.mark.parametrize("max_length", [128, 256, 512, 1024])
def test_toeplitz_matrix_rope(max_length: int):
    np.random.seed(0)

    for i in range(RANDOM_SAMPLES):
        a = np.random.rand(max_length)
        b = np.random.rand(max_length)

        tf_output: torch.Tensor = torch.tensor(
            tf_toeplitz_matrix_rope(
                max_length,
                tf.constant(a, dtype=tf.float32),
                tf.constant(b, dtype=tf.float32),
            ).numpy()
        )

        pt_output: torch.Tensor = toeplitz_matrix_rope(
            max_length,
            torch.tensor(a, dtype=torch.float32),
            torch.tensor(b, dtype=torch.float32),
        )

        assert_close(pt_output, tf_output, rtol=1e-3, atol=1e-3)
