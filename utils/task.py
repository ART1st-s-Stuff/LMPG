from abc import ABC, abstractmethod
from typing import Sequence

from utils.text import Text, text_window

class Agent(ABC):
    def __init__(self, instruction: str, tools: Sequence[Tool]):
        self.instruction = text_window(instruction)
        self.tools = tools
    
    @abstractmethod
    def terminate(self, output: str) -> bool:
        ...
        
    def step(self, output: str) -> str:
        if 