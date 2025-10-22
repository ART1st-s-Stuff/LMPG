import torch
from torch import nn

class QEvaluator(nn.Module):
    def __init__(self, state_dim):
        super(QEvaluator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value