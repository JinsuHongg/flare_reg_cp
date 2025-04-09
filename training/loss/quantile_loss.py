import torch
import torch.nn as nn 
from torch import Tensor

class QuantileLoss(nn.Module):
    """
    Parameters
    ----------
    1) target : 1d Tensor
    2) input : 1d Tensor, Predicted value.
    3) quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
        Quantileloss with quenatile of 0.5 is equal to mean absolute error
    """
    def __init__(self, quantile_val) -> None:
        super(QuantileLoss, self).__init__()
        self.quantile = quantile_val

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.shape != target.shape:
            print('Check input shape!')
            Print(f'target shape: {target.shape}, but input shape: {input.shape}')
        residual = torch.subtract(target, input)
        loss = torch.maximum(self.quantile * residual, (self.quantile - 1) * residual)
        return torch.mean(loss)