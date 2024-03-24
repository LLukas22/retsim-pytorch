import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List, Union
from enum import Enum


class PositionalEncodingType(Enum):
    """Enum for Positional Encoding types."""

    ROPE = "rope"
    RELATIVE = "relative"


class ScaledSinusoidalPositionalEmbedding(nn.Module):
    """Creates a positional embedding with a learnable scalar for stability.

    This layer calculates the position encoding as a mix of sine and cosine
    functions with geometrically increasing wavelengths. Defined and
    formulized in "Attention is All You Need", section 3.5.
    (https://arxiv.org/abs/1706.03762).
    """

    def __init__(
        self,
        hidden_size: int,
        min_timescale: float = 1.0,
        max_timescale: float = 1.0e4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.init_scale = 1 / self.hidden_size**0.5

        self.scale = nn.Parameter(torch.tensor(self.init_scale), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        length = inputs.shape[1]
        position = torch.arange(length, dtype=torch.float32).to(inputs.device)
        num_timescales = self.hidden_size // 2
        log_timescale_increment = torch.log(
            torch.tensor(
                float(self.max_timescale) / float(self.min_timescale),
                device=inputs.device,
            )
        ) / (torch.tensor(num_timescales, dtype=torch.float32) - 1)
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32, device=inputs.device)
            * -log_timescale_increment
        )
        scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(inv_timescales, 0)
        position_embeddings = torch.cat(
            (torch.sin(scaled_time), torch.cos(scaled_time)), dim=1
        )

        # scale position encodings with a learnable scalar
        position_embeddings *= self.scale

        return inputs + position_embeddings


def rope(x: torch.Tensor, axis: Union[List[int], int]) -> torch.Tensor:
    """RoPE positional encoding for PyTorch.

    Args:
        x: input tensor.
        axis: axis to add the positional encodings.

    Returns:
        The input tensor with RoPE encodings.
    """
    # Convert axis to a list if it's not already
    if isinstance(axis, int):
        axis = [axis]

    # Calculate the total length and create a position tensor
    spatial_shape = [x.size(i) for i in axis]
    total_len = torch.prod(torch.tensor(spatial_shape))
    position: torch.Tensor = (
        torch.arange(total_len, dtype=torch.float32).view(spatial_shape).to(x.device)
    )

    # Expand the position tensor along the necessary axes
    for i in range(axis[-1] + 1, x.dim() - 1):
        position = position.unsqueeze(-1)

    half_size = x.size(-1) // 2
    freq_seq = torch.arange(half_size, dtype=torch.float32) / float(half_size)
    inv_freq = 10000**-freq_seq
    sinusoid = torch.einsum("...,d->...d", position, inv_freq)
    sin = torch.sin(sinusoid).to(x.dtype)
    cos = torch.cos(sinusoid).to(x.dtype)

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def toeplitz_matrix_rope(n: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Obtain Toeplitz matrix using RoPE for PyTorch."""
    a = rope(torch.tile(a[None, :], (n, 1)), axis=0)
    b = rope(torch.tile(b[None, :], (n, 1)), axis=0)
    return torch.einsum("mk,nk->mn", a, b)


def toeplitz_matrix(n: int, w: torch.Tensor) -> torch.Tensor:
    """Toeplitz matrix for PyTorch."""
    # Create padding and repeat operations
    t = F.pad(w, (0, n))
    t = t.repeat(n)
    t = t[..., :-n]
    t = t.view(n, 3 * n - 2)
    r = (2 * n - 1) // 2
    return t[..., r:-r]


class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_len = max_len
        self.w = nn.Parameter(torch.randn(2 * self.max_len - 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = toeplitz_matrix(self.max_len, self.w)
        return x + bias


class RopePositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.a = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.b = nn.Parameter(torch.randn(hidden_size), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = toeplitz_matrix_rope(self.hidden_size, self.a, self.b)
        return x + bias


def get_positional_embedding_layer(
    max_len: int, type: PositionalEncodingType, **kwargs
) -> nn.Module:
    if type == PositionalEncodingType.ROPE:
        return RopePositionalEncoding(max_len, **kwargs)
    elif type == PositionalEncodingType.RELATIVE:
        return RelativePositionalEncoding(max_len, **kwargs)
    else:
        raise ValueError(f"Unsupported Positional Encoding type: {type}")
