import torch
from torch import nn

class Transition(nn.Module):
    dim_hidden = 128
    layers_ffnn = 6
    
    def __init__(self, dim_state: int, dim_action: int):
        super(Transition, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(dim_state + dim_action, self.dim_hidden),
            *[nn.Sequential(nn.ReLU(), nn.Linear(self.dim_hidden, self.dim_hidden)) for _ in range(self.layers_ffnn)],
            nn.ReLU(),
            nn.Linear(self.dim_hidden, dim_state)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.ffnn(x)