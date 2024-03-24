from src.retsim_pytorch.modules.metric_embedding import MetricEmbedding
import numpy as np
import torch
from torch.testing import assert_close


def test_is_constructable():
    pt_module = MetricEmbedding(in_features=256, out_features=256)

    assert pt_module is not None


def test_output_and_weights_match():
    import sys
    from unittest.mock import Mock

    sys.modules["nmslib"] = Mock()

    import tensorflow as tf
    from tensorflow_similarity.layers import MetricEmbedding as TFMetricEmbedding

    tf_modul = TFMetricEmbedding(256)
    pt_module = MetricEmbedding(in_features=256, out_features=256)

    # tf dense is lazy => we need to perform a forward pass to initialize the weights
    np.random.seed(0)
    inputs = np.random.rand(1, 512, 256)
    tf_modul.build(input_shape=(None, 512, 256))
    tf_result = tf_modul(tf.constant(inputs))

    weights = tf_modul.weights
    pt_weights = pt_module.state_dict()

    assert len(weights) == len(pt_weights)

    new_state_dict = {
        "weight": torch.tensor(weights[0].numpy().T),
        "bias": torch.tensor(weights[1].numpy()),
    }

    pt_module.load_state_dict(new_state_dict)

    for i in range(10):
        inputs = np.random.rand(1, 512, 256)
        tf_result = torch.tensor(
            tf_modul(tf.constant(inputs, dtype=tf.float32)).numpy()
        )
        pt_result = pt_module(torch.tensor(inputs, dtype=torch.float32))
        assert_close(tf_result, pt_result)
