import torch
import torch.nn as nn
from enum import Enum
from typing import Callable


class ActivationTypes(str, Enum):
    """Enum for Activation types."""

    RELU = "relu1"
    RELU2 = "relu2"
    SQRRELU = "sqrrelu"
    SWISH = "swish"
    GELU = "gelu"


class ReLU1(nn.Module):
    """ReLU1 activation function."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.clamp(inputs, min=0, max=1)


class ReLU2(nn.Module):
    """ReLU2 activation function."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.clamp(inputs, min=0, max=2)


class SqrReLU(nn.Module):
    """SqrReLU activation function."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return torch.relu(inputs) ** 2


def get_activation_layer(
    activation: ActivationTypes,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == ActivationTypes.RELU:
        return ReLU1()
    elif activation == ActivationTypes.RELU2:
        return ReLU2()
    elif activation == ActivationTypes.SQRRELU:
        return SqrReLU()
    elif activation == ActivationTypes.SWISH:
        return nn.SiLU()
    elif activation == ActivationTypes.GELU:
        return nn.GELU()
    else:
        return nn.Identity()
