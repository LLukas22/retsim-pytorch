import torch
import torch.nn as nn
from .modules import (
    GAU,
    GeneralizedMeanPooling1D,
    ScaledSinusoidalPositionalEmbedding,
    ActivationTypes,
    get_activation_layer,
    NormType,
    PositionalEncodingType,
    MetricEmbedding,
)
from dataclasses import dataclass
from typing import Optional


""" 
Defaults are from the layers of the original model.
Scaled_Sinusoidal_Positional_Embedding: {"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "dtype": "float32", "hidden_size": 24, "min_timescale": 1.0, "max_timescale": 10000.0}
GAU: {"name": "gau", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}
Generalized_Mean_Pooling_1D: {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "p": 3.0, "data_format": "channels_last", "keepdims": false}
"""


@dataclass
class RETSimConfig:
    binarizer_size: int = 24
    hidden_size: int = 256
    max_length: int = 512
    shared_dim: int = 128
    layers: int = 2
    dropout: float = 0.0
    min_timescale: int = 1
    max_timescale: int = 1e4
    activation: ActivationTypes = ActivationTypes.SWISH
    attention_activation: ActivationTypes = ActivationTypes.SQRRELU
    expansion_factor: int = 1
    norm: NormType = NormType.SCALED
    position_encoding: PositionalEncodingType = PositionalEncodingType.ROPE
    attention_dropout: float = 0.0
    spatial_dropout: float = 0.0
    epsilon: float = 1e-6
    p: float = 3.0


class RETSim(nn.Module):
    def __init__(self, config: Optional[RETSimConfig] = None) -> None:
        super().__init__()
        self.config = config or RETSimConfig()

        self.positional_embedding = ScaledSinusoidalPositionalEmbedding(
            hidden_size=self.config.binarizer_size,
            min_timescale=self.config.min_timescale,
            max_timescale=self.config.max_timescale,
        )

        self.encoder_start = nn.Linear(
            self.config.binarizer_size, self.config.hidden_size
        )

        self.activation = get_activation_layer(self.config.activation)

        gau_layers = []
        for i in range(self.config.layers):
            gau_layers.append(
                GAU(
                    dim=self.config.hidden_size,
                    max_len=self.config.max_length,
                    shared_dim=self.config.shared_dim,
                    expansion_factor=self.config.expansion_factor,
                    activation=self.config.activation,
                    attention_activation=self.config.attention_activation,
                    norm_type=self.config.norm,
                    position_encoding_type=self.config.position_encoding,
                    dropout_rate=self.config.dropout,
                    attention_dropout_rate=self.config.attention_dropout,
                    spatial_dropout_rate=self.config.spatial_dropout,
                    epsilon=self.config.epsilon,
                )
            )

        self.gau_layers = nn.ModuleList(gau_layers)

        self.mean_pooling = GeneralizedMeanPooling1D(p=self.config.p)

        self.dropout = nn.Dropout(self.config.dropout)

        self.metric_embedding = MetricEmbedding(
            in_features=self.config.hidden_size, out_features=self.config.hidden_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.positional_embedding(x)
        x = self.encoder_start(x)
        x = self.activation(x)

        for layer in self.gau_layers:
            x = layer(x)

        unpooled = x
        x = self.mean_pooling(x)
        x = self.dropout(x)
        x = self.metric_embedding(x)

        return x, unpooled
