from abc import ABC, abstractmethod
import re
from typing import List, Tuple, Callable, Literal, Dict, Any, TypeVar, Optional
from functools import wraps

import utils.settings as settings
from utils.scoring import Scoreboard

T = TypeVar('T', bound='Tool')
ToolFunction = Callable[[T, str, Scoreboard], str]
class Tool(ABC):
    @property
    @abstractmethod
    def interface(self) -> Dict[str, ToolFunction]:
        ...
        
    def invoke(self, tool_name: str, tool_input: str, scoreboard: Scoreboard) -> str:
        if tool_name not in self.interface:
            raise ToolNotExistException()
        return self.interface[tool_name](tool_input, scoreboard)

    def structurized_tool(self, type: Optional[Literal["json", "yaml", "toml"]] = None):
        if type is None:
            type = settings.STRUCTURIZED_TOOL_INPUT_FORMAT
        def wrapper(func: Callable[[...], Any]) -> ToolFunction:
            @wraps(func)
            def inner(tool_input: str, scoreboard: Scoreboard) -> str:
                match type:
                    case "json":
                        import json
                        kwargs = json.loads(tool_input)
                    case "yaml":
                        import yaml
                        kwargs = yaml.load(tool_input)
                    case "toml":
                        import toml
                        kwargs = toml.load(tool_input)
                    case _:
                        raise ValueError(f"Invalid type: {type}")
                kwargs["scoreboard"] = scoreboard
                return func(**kwargs)
            return inner
        return wrapper
    
class ToolCallException(Exception):
    def __init__(self, message: str, penalty: float):
        super().__init__("Tool Call Exception: " + message)
        self.penalty = penalty

class MultipleToolCallException(ToolCallException):
    def __init__(self):
        super().__init__("Calling more than 1 tools.", settings.INVALID_TOOL_CALL_PENALTY)

class MismatchedTagException(ToolCallException):
    def __init__(self):
        super().__init__("Mismatched tool call tags.", settings.INVALID_TOOL_CALL_PENALTY)

class ToolNotExistException(ToolCallException):
    def __init__(self):
        super().__init__("Tool does not exist.", settings.INVALID_TOOL_CALL_PENALTY)
        
class InvalidToolInputException(ToolCallException):
    def __init__(self, input: str, expected: str):
        super().__init__(f"Invalid tool input. Expected input: {expected}, Received: {input}", settings.INVALID_TOOL_CALL_PENALTY)

def parse_llm_output(output: str) -> Tuple[str, str, str]:
    regex = re.compile(r'<|(.*?)|>(.*)<|/(.*?)|>')
    match : List[str] = regex.findall(output)
    if len(match) > 1:
        raise MultipleToolCallException()
    tool_call = match[0]
    if tool_call[0] != tool_call[2]:
        raise MismatchedTagException()
    tag = tool_call[0].split(":")
    return tag[0], tag[1], tool_call[1]  # context, tool_name, tool_input