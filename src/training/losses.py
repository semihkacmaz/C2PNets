import torch
import torch.nn as nn

class PenalizedMSELoss(nn.Module):
    def __init__(self, penalty_factor=10.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.penalty_factor = penalty_factor

    def forward(self, predictions, targets, mean=0, std=0):
        mse_loss = self.mse(predictions, targets)
        penalty = torch.relu(-(predictions * std + mean)).sum()
        return mse_loss + self.penalty_factor * penalty
