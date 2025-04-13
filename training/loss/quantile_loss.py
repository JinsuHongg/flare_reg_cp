import torch
import torch.nn as nn 
from torch import Tensor

class QuantileLoss(nn.Module):
    def __init__(self, quantiles: list) -> None:
        super(QuantileLoss, self).__init__()
        self.quantile_lst = quantiles

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.device == target.device, f"Device mismatch! input: {input.device}, target: {target.device}"
        
        if input.shape[0] != target.shape[0]:
            raise ValueError(f"Shape mismatch: target shape {target.shape}, input shape {input[:, 0].shape}")
        
        loss = []
        for index, quantile in enumerate(self.quantile_lst):
            pred = input[:, index].unsqueeze(1)
            residual = target.unsqueeze(1) - pred
            loss_i = torch.maximum(quantile * residual, (quantile - 1) * residual)
            loss.append(loss_i)
            # print(torch.cat([pred, target.unsqueeze(1), loss_i], dim=1))
            # print(loss_i)
        # Stack & sum across quantiles, then mean across batch
        total_loss = torch.mean(torch.sum(torch.cat(loss, dim=1), dim=1))
        # print(torch.sum(torch.cat(loss, dim=1), dim=1))
        # print(torch.cat([input, target.unsqueeze(1), torch.cat(loss, dim=1), torch.sum(torch.cat(loss, dim=1), dim=1).unsqueeze(1)], dim=1))
        return total_loss
