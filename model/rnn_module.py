from typing import List

import torch
from torch import nn

class RNNBlock(nn.Module):
    # Currently single head
    def __init__(self, dim_input: int, dim_hidden: int, num_layers: int, dim_output: int, dropout: float = 0.1):
        super().__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(dim_output + dim_input, dim_hidden),
            *[nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.Dropout(dropout)) for _ in range(num_layers - 1)],
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_output)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim_output + dim_input, dim_hidden),
            nn.Linear(dim_hidden, 1),
            nn.Tanh()
        )
        #self.initial_state = nn.Parameter(torch.zeros(1, 1, dim_output))
        self.zeros = torch.zeros(dim_output)
        self._dim_output = dim_output
        
    def forward(self, x, state):
        input = torch.cat([x, state], dim=-1)
        current_state = self.ffnn(input)
        gate = self.gate(input)
        next_state = current_state * gate + state * (1 - gate)
        # Gate=0: keep old state, Gate=1: take new state
        return next_state
    
    #def initialize(self, initial_state: torch.Tensor, batch_size: int):
    #    self.initial_state = nn.Parameter(initial_state.expand(1, batch_size, -1))
    #    return self.initial_state
    
    @property
    def dim_output(self):
        return self._dim_output
    
    def get_initial_state(self):
        return self.zeros
    
class StackedRNNBlock(nn.Module):
    def __init__(self, blocks: List[RNNBlock]):
        super().__init__()
        self._blocks = blocks
        self.blocks = nn.ModuleList(blocks)
        self._dim_states = [b.dim_output for b in blocks]
        self._dim_output = sum(self._dim_states)
        self._state_slices = []
        i = 0
        for d in self._dim_states:
            self._state_slices.append(slice(i, i + d))
            i += d

    def forward(self, x, states):
        new_states = []
        input = x
        for block, state_slice in zip(self.blocks, self._state_slices):
            state = states[state_slice]
            new_state = block(input, state)
            input = new_state
            new_states.append(new_state)
        return torch.cat(new_states)
    
    @property
    def dim_output(self) -> int:
        return self._dim_output

    def get_initial_states(self, device: torch.device):
        return torch.cat([block.get_initial_state() for block in self._blocks]).to(device)