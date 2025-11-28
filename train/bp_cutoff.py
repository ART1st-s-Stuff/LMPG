from abc import ABC, abstractmethod
from typing import Literal
from collections import deque

import torch

# TODO: Implement this using .detach_()

class TraceManager:
    """
    Trace manager cuts off the BP trace to save memory.
    """

    def __init__(self, strategy: Literal["by_tokens", "by_steps", "disabled"], max_length: int = -1):
        self.instance = 

        if strategy == "disabled":
            self.max_length = -1
        else:
            if max_length <= 0:
                raise ValueError(f"max_length must be greater than 0, got {max_length}")
            if self.strategy == "by_steps":
                self.queue = deque(maxlen=max_length)

    def cutoff(self) -> torch.Tensor:
        if len(self.queue) >= self.max_length:
            return self.queue.popleft()
        return trace

class _ITraceManager(ABC):
    @abstractmethod
    def cutoff(self, input: torch.Tensor) -> None:
        raise NotImplementedError()

class _TraceManagerByTokens(_ITraceManager):
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.history = torch.tensor([])

    def cutoff(self) -> torch.Tensor:
        self.history = torch.cat([self.history, input], dim=0)
        if self.history.shape[0] >= self.max_length:
            self.history = self.history[self.history.shape[0] - self.max_length:]
            self.history[0]


class _TraceManagerBySteps(TraceManager):
    def __init__(self, max_length: int = -1):
        self.max_length = max_length

    def cutoff(self) -> torch.Tensor:
        if len(self.queue) >= self.max_length:
            return self.queue.popleft()