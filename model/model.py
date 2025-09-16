from typing import Callable, List, Optional

import torch
from torch import nn

from model.rnn_module import StackedRNNBlock, RNNBlock

class HRNNEncoder(nn.Module):
    def __init__(self, encoder: StackedRNNBlock, dim_hidden: int, dim_output: int):
        super().__init__()
        self.encoder = encoder
        self.ffnn = nn.Sequential(
            nn.Linear(dim_output, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_output)
        )
        self._dim_output = dim_output

    def forward(self, states, x):
        states = self.encoder(states, x)
        output = self.ffnn(torch.cat(states, dim=-1))
        return output, states
    
    @property
    def dim_output(self):
        return self._dim_output

    @staticmethod
    def default():
        encoder_rnn_block_args = {
            'dim_hidden': 2048,
            'dropout': 0.3,
            'num_layers': 3
        }
        dim_output_middle = 2048
        return HRNNEncoder(
            StackedRNNBlock(
                blocks=[
                    RNNBlock(
                        dim_input=32,       # UTF-32
                        dim_output=dim_output_middle,
                        **encoder_rnn_block_args
                    ),
                    RNNBlock(
                        dim_input=dim_output_middle,
                        dim_output=dim_output_middle,
                        **encoder_rnn_block_args
                    ),
                    RNNBlock(
                        dim_input=dim_output_middle,
                        dim_output=dim_output_middle,
                        **encoder_rnn_block_args
                    ),
                    RNNBlock(
                        dim_input=dim_output_middle,
                        dim_output=dim_output_middle,
                        **encoder_rnn_block_args
                    ),
                ]
            ), dim_hidden=dim_output_middle * 4, dim_output=512)
    
class HRNN(nn.Module):
    def __init__(self, encoders: List[HRNNEncoder], decoder: RNNBlock):
        super().__init__()
        self.encoders = encoders
        self.decoder = decoder
        
    def forward(self, x: torch.Tensor, states=None, post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        if states is None:
            states = self.get_initial_states(x.device)
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            new_states = [self.encoders[i](states[i], x_t) for i in range(len(self.encoders))]
            new_states_tensor = torch.cat([ns[1] for ns in new_states], dim=0)
            y = self.decoder(new_states_tensor)
            processed_y = post_processing(y) if post_processing is not None else y
            output = processed_y if t == 0 else torch.cat([output, processed_y], dim=1)
            states = new_states
        return output
    
    def self_regression(self, x: torch.Tensor, max_output: int, halt_if: Callable[[torch.Tensor], bool], states=None, post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        output = self.forward(x, states)
        while output.size(1) < max_output:
            x_t = output[:, -1, :].unsqueeze(1)
            y_t = self.forward(x_t, states)
            output = torch.cat([output, y_t], dim=1)
            if halt_if(y_t):
                break
        return output

    @staticmethod
    def default():
        encoders = [ HRNNEncoder.default() for _ in range(16) ]
        decoder = RNNBlock(
            dim_input=encoders[-1].dim_output * len(encoders),
            dim_hidden=2048,
            dim_output=1024,
            num_layers=6
        )
        return HRNN(encoders, decoder)
    
    @property
    def dim_output(self):
        return self.decoder.dim_output
    
    def get_initial_states(self, device: torch.device):
        return [ [torch.zeros(1, encoder.dim_output, device=device) for _ in range(len(encoder.encoder.blocks))] for encoder in self.encoders ]