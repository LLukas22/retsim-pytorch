import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GeneralizedMeanPooling1D(nn.Module):
    def __init__(
        self, p: float = 3.0, step_axis: int = 1, keepdims: bool = False
    ):
        super(GeneralizedMeanPooling1D, self).__init__()
        self.p = p
        self.step_axis = step_axis
        self.keepdims = keepdims

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        mins = torch.min(x, dim=self.step_axis, keepdim=True)[0]
        x_offset = x - mins + 1
        
        if abs(self.p) < 0.00001:
            x_offset = self._geometric_mean(x_offset)
        elif self.p == math.inf:
            x_offset = self._pos_inf(x_offset)
        elif self.p == -math.inf:
            x_offset = self._neg_inf(x_offset)
        else:
            x_offset = self._generalized_mean(x_offset)
            
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
        
