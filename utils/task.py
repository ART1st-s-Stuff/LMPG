from abc import ABC, abstractmethod
from typing import Union
import re

import torch

from utils.scoring import Scoreboard
from utils.environment import Environment
import utils.settings as settings

class Agent(ABC):
    def __init__(self, environment: Environment):
        self.environment = environment
        self.scoreboard = environment.scoreboard
        self.history = 
    
    @abstractmethod
    def pause(self):
        ...
    
    @abstractmethod
    def invoke(self, input: torch.Tensor) -> str:
        ...
        
    def step(self, output: str) -> str:
        # Parse output
        regex = re.compile(r'<|(.*?)|>(.*)</|(.*?)|>')
        match = regex.findall(output)
        if len(match) > 1:
            self.scoreboard.reward(settings.INVALID_TOOL_CALL_PENALTY, "More than 1 tool called")
            return ""
        elif len(match) == 1:
            # Call tools
            tool_name = match[0]
    
    def interrupt(self, input: str):
        self.pause()
        self.invoke(input)