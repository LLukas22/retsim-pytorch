import sys
from unittest.mock import Mock

sys.modules["nmslib"] = Mock()

from retvec.tf.models.gau import GAU as TFGAU
from retvec.tf.models.positional_embeddings import (
    toeplitz_matrix_rope as tf_toeplitz_matrix_rope,
)
from src.retsim_pytorch.modules.gau import GAU
import numpy as np
import torch
import tensorflow as tf
import pytest
from torch.testing import assert_close


def test_is_constructable():
    pt_module = GAU(dim=256)
    assert pt_module is not None

def test_is_compilable():
    pt_module = GAU(dim=256, shared_dim=128, max_len=512)
    inputs = torch.rand(1, 512, 256)

    pt_module = torch.jit.script(pt_module)
    pt_output = pt_module(inputs)

    assert pt_output is not None

def test_output_sizes_match():
    tf_module = TFGAU(dim=256, shared_dim=128, max_len=512, expansion_factor=2)
    pt_module = GAU(dim=256, shared_dim=128, max_len=512, expansion_factor=2)

    # Test the forward method
    inputs = np.random.rand(1, 512, 256)

    tf_output: tf.Tensor = tf_module(tf.constant(inputs))
    pt_output: torch.Tensor = pt_module(torch.tensor(inputs, dtype=torch.float32))

    assert tf_output.shape == pt_output.shape


def test_weight_shapes_match():
    tf_module = TFGAU(dim=256, shared_dim=128, max_len=512, expansion_factor=2)
    pt_module = GAU(dim=256, shared_dim=128, max_len=512, expansion_factor=2)

    # tf dense is lazy => we need to perform a forward pass to initialize the weights
    inputs = np.random.rand(1, 512, 256)
    tf_module.build(input_shape=(None, 512, 256))
    tf_module(tf.constant(inputs))

    tf_weights = tf_module.weights
    pt_weights = pt_module.state_dict()

    assert len(tf_weights) == len(pt_weights)
    assert tf_weights[0].shape == pt_weights["positional_encoding.a"].shape
    assert tf_weights[1].shape == pt_weights["positional_encoding.b"].shape
    assert tf_weights[2].shape == pt_weights["gamma"].shape
    assert tf_weights[3].shape == pt_weights["beta"].shape
    assert tf_weights[4].shape == pt_weights["norm.scale"].shape
    assert tf.transpose(tf_weights[5]).shape == pt_weights["proj1.weight"].shape
    assert tf_weights[6].shape == pt_weights["proj1.bias"].shape
    assert tf.transpose(tf_weights[7]).shape == pt_weights["proj2.weight"].shape
    assert tf_weights[8].shape == pt_weights["proj2.bias"].shape


def test_can_load_tf_weights():
    tf_module = TFGAU(dim=256, shared_dim=128, max_len=512, expansion_factor=1)
    pt_module = GAU(dim=256, shared_dim=128, max_len=512, expansion_factor=1)

    np.random.seed(0)
    # tf dense is lazy => we need to perform a forward pass to initialize the weights
    inputs = np.random.rand(1, 512, 256)
    tf_module.build(input_shape=(None, 512, 256))
    tf_result = tf_module(tf.constant(inputs, dtype=tf.float32))

    tf_weights = tf_module.weights

    state_dict = {
        "positional_encoding.a": torch.tensor(tf_weights[0].numpy()),
        "positional_encoding.b": torch.tensor(tf_weights[1].numpy()),
        "gamma": torch.tensor(tf_weights[2].numpy()),
        "beta": torch.tensor(tf_weights[3].numpy()),
        "norm.scale": torch.tensor(tf_weights[4].numpy()),
        "proj1.weight": torch.tensor(tf.transpose(tf_weights[5]).numpy()),
        "proj1.bias": torch.tensor(tf_weights[6].numpy()),
        "proj2.weight": torch.tensor(tf.transpose(tf_weights[7]).numpy()),
        "proj2.bias": torch.tensor(tf_weights[8].numpy()),
    }

    pt_module.load_state_dict(state_dict)
    pt_module = pt_module.eval()

    for i in range(10):
        inputs = np.random.rand(1, 512, 256)
        assert_close(
            pt_module(torch.tensor(inputs, dtype=torch.float32)),
            torch.tensor(tf_module(tf.constant(inputs, dtype=tf.float32)).numpy()),
        )


def test_is_onnx_exportable():
    pt_module = GAU(dim=256, shared_dim=128, max_len=512, expansion_factor=1)
    inputs = torch.rand(1, 512, 256)

    pt_module = torch.jit.script(pt_module)
    pt_module.eval()
    pt_output = pt_module(inputs)

    torch.onnx.export(
        pt_module,
        inputs,
        "gau.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}},
    )

    import onnx
    onnx.checker.check_model(onnx.load("gau.onnx"),full_check=True)