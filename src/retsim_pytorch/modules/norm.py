import torch
import torch.nn as nn
from enum import Enum


class NormType(Enum):
    """Enum for Norm types."""

    LAYER = "layer"
    BATCH = "batch"
    L2 = "l2"
    SCALED = "scaled"


class ScaledNorm(nn.Module):
    """ScaledNorm layer."""

    def __init__(self, begin_axis: int = -1, epsilon: float = 1e-5):
        """Initialize a ScaledNorm Layer.

        Args:
            begin_axis: Axis along which to apply norm. Defaults to -1.

            epsilon: Norm epsilon value. Defaults to 1e-5.
        """
        super(ScaledNorm, self).__init__()
        self.begin_axis = begin_axis
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.tensor(1.0))  # Learnable scale parameter

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        # Compute the mean square across the specified axes
        dims = list(range(len(x.shape)))[self.begin_axis :]
        mean_square = torch.mean(x**2, dim=dims, keepdim=True)
        # Apply normalization
        x = x * torch.rsqrt(mean_square + self.epsilon)
        return x * self.scale

    def extra_repr(self) -> str:
        """Set the extra representation of the module, which will come in handy during debugging."""
        return f"begin_axis={self.begin_axis}, epsilon={self.epsilon}"


def get_norm_layer(norm_type: NormType, **kwargs) -> nn.Module:
    """Get the normalization layer.

    Args:
        norm_type: Type of normalization layer.

        epsilon: Norm epsilon value.

    Returns:
        Normalization layer.
    """
    if norm_type == NormType.LAYER:
        return nn.LayerNorm(**kwargs)
    elif norm_type == NormType.BATCH:
        return nn.BatchNorm1d(**kwargs)
    elif norm_type == NormType.L2:
        return nn.LayerNorm(**kwargs)
    elif norm_type == NormType.SCALED:
        return ScaledNorm(**kwargs)
    else:
        raise ValueError(f"Invalid norm type: {norm_type}")
