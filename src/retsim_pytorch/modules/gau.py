import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import get_activation_layer, ActivationTypes
from .positional_embeddings import (
    get_positional_embedding_layer,
    PositionalEncodingType,
)
from .norm import get_norm_layer, NormType


class GAU(nn.Module):
    """Gated Attention Unit layer introduced in Transformer Quality in Linear Time.
    Paper reference: https://arxiv.org/abs/2202.10447
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 128,
        shared_dim: int = 128,
        expansion_factor: int = 2,
        activation: ActivationTypes = ActivationTypes.SWISH,
        attention_activation: ActivationTypes = ActivationTypes.SQRRELU,
        norm_type: NormType = NormType.SCALED,
        position_encoding_type: PositionalEncodingType = PositionalEncodingType.ROPE,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        spatial_dropout_rate: float = 0.0,
        epsilon: float = 1e-6,
    ):
        """
        Initialize a GAU layer.

        Args:
            dim: Dimension of GAU block.

            max_len: Maximum seq len of input. Defaults to 128.

            shared_dim: Size of shared dim. Defaults to 128.

            expansion_factor: Hidden dim expansion factor. Defaults to 2.

            activation: Activation to use in projection layers. Defaults
                to 'ActivationTypes.SWISH'.

            attention_activation: Activation to use on attention scores.
                Defaults to 'ActivationTypes.SQRRELU'.

            norm_type: Norm type.
                Defaults to 'NormType.SCALED'.

            position_encoding_type: Type of positional encoding to use.
                Defaults to 'PositionalEncodingType.ROPE'.

            dropout_rate: Feature dropout rate. Defaults to 0.0.

            attention_dropout_rate: Attention dropout rate.
                Defaults to 0.0.

            spatial_dropout_rate: Spatial dropout rate. Defaults to 0.0.

            epsilon: Epsilon value for norm. Defaults to 1e-6.
        """
        super(GAU, self).__init__()
        self.dim = dim
        self.max_len = max_len
        self.shared_dim = shared_dim
        self.expansion_factor = expansion_factor
        self.activation = get_activation_layer(activation)
        self.attention_activation = get_activation_layer(attention_activation)
        self.norm = get_norm_layer(norm_type, epsilon=epsilon)

        self.position_encoding_type = position_encoding_type
        self.dropout_rate = dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.epsilon = epsilon

        self.expand_dim = self.dim * self.expansion_factor
        self.proj_dim = 2 * self.expand_dim + self.shared_dim

        # Define layers
        self.proj1 = nn.Linear(self.dim, self.proj_dim, bias=True)
        self.proj2 = nn.Linear(self.expand_dim, self.dim, bias=True)

        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.attention_dropout = nn.Dropout(self.attention_dropout_rate)

        # Positional encodings and related parameters
        self.positional_encoding = get_positional_embedding_layer(
            self.max_len, self.position_encoding_type
        )

        # Offset scaling values
        self.gamma = nn.Parameter(torch.empty(2, self.shared_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(2, self.shared_dim), requires_grad=True)
        nn.init.normal_(self.gamma, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm(x)

        # Input dropout
        if self.spatial_dropout_rate > 0:
            # Implementing spatial dropout in PyTorch
            x = (
                F.dropout2d(
                    x.permute(0, 2, 1).unsqueeze(3),
                    self.spatial_dropout_rate,
                    self.training,
                )
                .squeeze(3)
                .permute(0, 2, 1)
            )

        x = self.dropout1(x)

        # Initial projection to generate uv
        uv = self.proj1(x)
        uv = self.activation(uv)

        uv = self.dropout2(uv)
        u, v, base = torch.split(
            uv, [self.expand_dim, self.expand_dim, self.shared_dim], dim=-1
        )

        # Generate q, k by scaled offset
        base_expanded = (
            torch.einsum("bnr,hr->bnhr", base, self.gamma) + self.beta
        )  # e.g. (1, 512, 2, 256)
        q, k = base_expanded.unbind(dim=2)  # e.g. (1, 512, 256)

        # Compute key-query scores
        qk = torch.einsum(
            "bnd,bmd->bnm", q, k
        )  # e.g. (1, 512, 512)
        qk = qk / self.max_len

        qk = self.positional_encoding(qk)

        # Apply attention activation and dropout
        kernel = self.attention_activation(qk)
        if self.attention_dropout_rate > 0:
            kernel = self.attention_dropout(kernel)

        # Apply values and project
        x = u * torch.einsum("bnm,bme->bne", kernel, v)  # e.g. (1, 512, 512)
        x = self.proj2(x)  # e.g. (1, 512, 256)
        return x + shortcut
