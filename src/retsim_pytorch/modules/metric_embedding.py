import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricEmbedding(nn.Module):
    """L2 Normalized `Linear` layer.

    This layer is usually used as an output layer, especially when using cosine
    distance as the similarity metric.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MetricEmbedding, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.linear(inputs)
        normed_x = F.normalize(x, p=2.0, dim=1)  # L2 normalize across feature dimension
        return normed_x
