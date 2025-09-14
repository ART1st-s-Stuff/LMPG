from abc import ABC, abstractmethod
from typing import TypedDict

class InteractionResult(TypedDict):
    output: str
    time: float

class InteractionProvider(ABC):
    @abstractmethod
    def interact(self, action: str) -> InteractionResult:
        pass