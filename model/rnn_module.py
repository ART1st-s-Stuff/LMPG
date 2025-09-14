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
            nn.Linear(dim_output, dim_hidden),
            nn.Linear(dim_hidden, 1),
            nn.Tanh()
        )
        #self.initial_state = nn.Parameter(torch.zeros(1, 1, dim_output))
        self.zeros = torch.zeros(1, 1, dim_output)
        self._dim_output = dim_output
        
    def forward(self, state, x):
        next_state = self.ffnn(torch.cat([state, x], dim=-1))
        gate = self.gate(next_state)
        next_state = next_state * gate + state * (1 - gate)
        # Gate=0: keep old state, Gate=1: take new state
        return next_state, gate
    
    #def initialize(self, initial_state: torch.Tensor, batch_size: int):
    #    self.initial_state = nn.Parameter(initial_state.expand(1, batch_size, -1))
    #    return self.initial_state
    
    @property
    def dim_output(self):
        return self._dim_output
    
class StackedRNNBlock(nn.Module):
    def __init__(self, blocks: List[RNNBlock], parallel: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self._dim_output = sum([b.dim_output for b in blocks])
        self.parallel = parallel

    def forward(self, state_list, x):
        new_states = []
        input = x
        for block, state in zip(self.blocks, state_list):
            new_state, gate = block(state, input)
            input = new_state
            # Gate=0: keep old state, Gate=1: take new state
            if self.parallel:
                new_states.append(state * (1 - gate) + block.initial_state * gate)
            else:
                new_states.append(state * (1 - gate) + new_state * gate)
        return new_states
    
    @property
    def dim_output(self) -> int:
        return self._dim_output
