import keras
import torch
from pathlib import Path


def process_linear(weights, bias, prefix: str) -> dict[str, torch.Tensor]:
    weight_map = {}

    transformed_weights = torch.tensor(weights.numpy()).T.contiguous()
    bias = torch.tensor(bias.numpy())

    weight_map[prefix + ".weight"] = transformed_weights
    weight_map[prefix + ".bias"] = bias

    return weight_map


def process_gau(layers, prefix: str) -> dict[str, torch.Tensor]:
    weight_map = {
        f"{prefix}.positional_encoding.a": torch.tensor(layers[0].numpy()),
        f"{prefix}.positional_encoding.b": torch.tensor(layers[1].numpy()),
        f"{prefix}.gamma": torch.tensor(layers[2].numpy()),
        f"{prefix}.beta": torch.tensor(layers[3].numpy()),
        f"{prefix}.norm.scale": torch.tensor(layers[4].numpy()),
    }

    proj1 = process_linear(*layers[5:7], prefix=f"{prefix}.proj1")
    proj2 = process_linear(*layers[7:], prefix=f"{prefix}.proj2")

    weight_map.update(proj1)
    weight_map.update(proj2)

    return weight_map


def convert_retsim(directory: Path) -> dict[str, torch.Tensor]:
    """
    Convert a keras model to a pytorch state_dict
    """
    keras_model: keras.models.Model = keras.models.load_model(directory, compile=False)
    weights = keras_model.weights

    positional_scale = {"positional_embedding.scale": torch.tensor(weights[0].numpy())}
    encoder_start = process_linear(*weights[1:3], prefix="encoder_start")
    gau_0_weights = process_gau(weights[3:12], prefix="gau_layers.0")
    gau_1_weights = process_gau(weights[12:21], prefix="gau_layers.1")
    metric_embedding = process_linear(*weights[21:], prefix="metric_embedding.linear")
    weight_map = {
        **positional_scale,
        **encoder_start,
        **gau_0_weights,
        **gau_1_weights,
        **metric_embedding,
    }

    return weight_map
