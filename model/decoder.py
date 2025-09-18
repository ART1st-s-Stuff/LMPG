from typing import Literal

import torch
from torch import nn

class FFNNDecoder(nn.Module):
    def __init__(self, dim_input: int, dim_hidden: int, num_layers: int, dim_output: int, dropout: float = 0.3, type: Literal["softmax", "sigmoid"] = "softmax"):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            *[nn.Sequential(nn.GELU(), nn.Linear(dim_hidden, dim_hidden), nn.Dropout(dropout)) for _ in range(num_layers - 1)],
            nn.GELU(),
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=-1) if type == "softmax" else nn.Sigmoid()
        )

    def forward(self, x):
        return self.ffnn(x)