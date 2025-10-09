from torch import nn

class Eval(nn.Module):
    dim_hidden = 128
    layers_ffnn = 6
    
    def __init__(self, dim_state: int, dim_value: int):
        super(Eval, self).__init__()
        self.ffnn = nn.Sequential(
            nn.Linear(dim_state, self.dim_hidden),
            *[nn.Sequential(nn.ReLU(), nn.Linear(self.dim_hidden, self.dim_hidden)) for _ in range(self.layers_ffnn)],
            nn.ReLU(),
            nn.Linear(self.dim_hidden, dim_value)
        )
        
    def forward(self, x):
        return self.ffnn(x)