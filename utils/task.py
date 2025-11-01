from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from utils.scoring import Scoreboard
from utils.environment import Environment
from utils.tool import Tool, parse_llm_output, ToolCallException, ToolNotExistException
from utils.text import TextWindow
import utils.settings as settings

class Agent(ABC):
    def __init__(self, environment: Environment):
        self.environment = environment
        self.scoreboard = environment.scoreboard
        self.contexts : Dict[str, Tool] = {}
    
    @abstractmethod
    def pause(self):
        ...
    
    @abstractmethod
    def invoke(self, input: torch.Tensor) -> str:
        ...
        
    def step(self, output: str) -> str:
        # Parse output
        try:
            context, tool, tool_input = parse_llm_output(output)
            if context not in self.contexts:
                raise ToolNotExistException()
            ctx = self.contexts[context]
            tool_output = ctx.invoke(tool, tool_input)
        except ToolCallException as e:
            self.scoreboard.reward(e.penalty, str(e))
            return ""
        return tool_output

    def interrupt(self, input: str):
        self.pause()
        self.invoke(input)