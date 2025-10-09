from typing import Callable, List, Optional

import torch
from torch import nn

from model.rnn_module import StackedRNNBlock, RNNBlock

class HRNNEncoder(nn.Module):
    def __init__(self, encoder: StackedRNNBlock, dim_hidden: int, dim_output: int):
        super().__init__()
        self.encoder = encoder
        self.ffnn = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, dim_output)
        )
        self._dim_output = dim_output

    def forward(self, x, states):
        states = self.encoder(x, states)
        output = self.ffnn(states)
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
            ), dim_hidden=dim_output_middle * 4, dim_output=1024)
        
    @staticmethod
    def tiny():
        encoder_rnn_block_args = {
            'dim_hidden': 256,
            'dropout': 0.3,
            'num_layers': 3
        }
        dim_output_middle = 256
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
            ), dim_hidden=dim_output_middle * 4, dim_output=256)
    
class HRNN(nn.Module):
    def __init__(self, encoders: List[HRNNEncoder], decoder: RNNBlock):
        super().__init__()
        self._encoders = encoders
        self.encoders = nn.ModuleList(encoders)
        
    def forward(self, x: torch.Tensor, states=None, post_processing: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        if states is None:
            states = self.get_initial_states(x.device)
        for t in range(x.size(0)):
            x_t = x[t, :]
            encoded = [self.encoders[i](x_t, states[i]) for i in range(len(self.encoders))]
            ns_list = []
            y_list = []
            for y, ns in encoded:
                ns_list.append(ns)
                y_list.append(y)
            y_tensor = torch.cat(y_list)
            processed_y = post_processing(y_tensor) if post_processing is not None else y_tensor
            if processed_y.dim() == 1:
                processed_y = processed_y.unsqueeze(0)
            output = processed_y if t == 0 else torch.cat([output, processed_y], dim=1)
            states = ns_list
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
            dim_hidden=4096,
            dim_output=1024,
            num_layers=6
        )
        return HRNN(encoders, decoder)
    
    @staticmethod
    def tiny():
        encoders = [ HRNNEncoder.tiny() for _ in range(8) ]
        decoder = RNNBlock(
            dim_input=encoders[-1].dim_output * len(encoders),
            dim_hidden=2048,
            dim_output=512,
            num_layers=6
        )
        return HRNN(encoders, decoder)
    
    @property
    def dim_output(self):
        return sum([encoder.dim_output for encoder in self._encoders])
    def get_initial_states(self, device: torch.device):
        return [ encoder.encoder.get_initial_states(device) for encoder in self._encoders ]