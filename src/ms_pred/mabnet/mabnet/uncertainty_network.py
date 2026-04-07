
import torch
import torch.nn as nn
from edl_pytorch import NormalInvGamma


class UncertaintyNetwork(nn.Module):
    def __init__(self, hidden_channels, activation="relu"):
        super(UncertaintyNetwork, self).__init__()
        self.activation = activation

        self.nn_module = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.ReLU() if activation == "relu" else nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU() if activation == "relu" else nn.SiLU(),
            NormalInvGamma(hidden_channels // 2, 1),  # Output layer for uncertainty prediction
        )

    def forward(self, x):
        return self.nn_module(x)