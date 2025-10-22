import torch
from torch import nn

class ProgressiveWeights(torch.optim.Optimizer):
    def __init__(self, params,
                 dtype : torch.dtype = torch.float16,
                 lr : float = 0.001,
                 gate_forget : float = 0.001,
                 gate_learn : float = 0.001):
        super().__init__(params, {"lr": lr, "gate_forget": gate_forget, "gate_learn": gate_learn})
        self.dtype = dtype

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                l_weight = state.get('l_weight', torch.zeros_like(p, self.dtype))
                l_weight *= 1 - group['gate_forget']
                l_weight += grad.to(self.dtype) * group['lr'] * (1 - group['gate_learn'])
                p += grad * group['lr'] * group['gate_learn']
                