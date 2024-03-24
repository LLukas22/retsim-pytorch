import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeneralizedMeanPooling(nn.Module):
    def __init__(
        self, p: float = 3.0, data_format: str = "channels_last", keepdims: bool = False
    ):
        super(GeneralizedMeanPooling, self).__init__()
        self.p = p
        self.data_format = data_format
        self.keepdims = keepdims
        self.step_axis = 1 if data_format == "channels_last" else 2

        if abs(self.p) < 0.00001:
            self.compute_mean = self._geometric_mean
        elif self.p == math.inf:
            self.compute_mean = self._pos_inf
        elif self.p == -math.inf:
            self.compute_mean = self._neg_inf
        else:
            self.compute_mean = self._generalized_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            mins = torch.min(x, dim=self.step_axis, keepdim=True)[0]
            x_offset = x - mins + 1
        else:
            mins = torch.min(x, dim=self.step_axis, keepdim=True)[0]
            x_offset = x - mins + 1

        x_offset = self.compute_mean(x_offset)
        x = x_offset + mins - 1

        if not self.keepdims:
            x = x.squeeze(self.step_axis)
        return x

    def _geometric_mean(self, x: torch.Tensor):
        x = torch.log(x)
        x = torch.mean(x, dim=self.step_axis, keepdim=True)
        return torch.exp(x)

    def _generalized_mean(self, x: torch.Tensor):
        x = x.pow(self.p)
        x = torch.mean(x, dim=self.step_axis, keepdim=True)
        return x.pow(1.0 / self.p)

    def _pos_inf(self, x: torch.Tensor):
        return torch.max(x, dim=self.step_axis, keepdim=True)[0]

    def _neg_inf(self, x: torch.Tensor):
        # Implement as negative of the positive inf of -x
        return -self._pos_inf(-x)


class GeneralizedMeanPooling1D(GeneralizedMeanPooling):
    def __init__(
        self, p: float = 3.0, data_format: str = "channels_last", keepdims: bool = False
    ):
        super(GeneralizedMeanPooling1D, self).__init__(
            p=p, data_format=data_format, keepdims=keepdims
        )
