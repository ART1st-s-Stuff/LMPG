from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, TypeVar

class Tool(ABC):
    def __init__(self, scoreboard):
        self.scoreboard = scoreboard
    
    @abstractmethod
    def interface(self):
        ...
    
def parse_llm_output(output: str) -> str:
    ...