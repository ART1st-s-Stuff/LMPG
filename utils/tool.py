from abc import ABC
import re
from typing import List, Tuple, Callable, Literal, Any, TypeVar, Optional, Dict, NamedTuple
from functools import wraps
import json

from . import settings
from .scoring import Scoreboard
from .exceptions import ToolNotExistException, MultipleToolCallException, InvalidToolException, InvalidToolCallJSONException

T = TypeVar('T', bound='Toolset')
ToolFunction = Callable[[T, str, str, Scoreboard], str]
class Toolset(ABC):
    interface : Dict[str, 'Toolset._ToolItem'] = {}
    
    class _ToolItem(NamedTuple):
        prompt : str
        func : ToolFunction

    def invoke(self, context: str, tool_name: str, tool_input: str, _scoreboard: Scoreboard) -> str:
        if tool_name not in self.interface:
            raise ToolNotExistException(context, tool_name)
        return self.interface[tool_name].func(self, context, tool_input, _scoreboard)

    @staticmethod
    def tool(tool_name: Optional[str] = None, description: Optional[str] = None) -> Callable[[ToolFunction], ToolFunction]:
        def wrapper(func: ToolFunction):
            @wraps(func)
            def inner(self, context: str, tool_input: str, _scoreboard: Scoreboard) -> str:
                name = tool_name or func.__name__
                desc = description or func.__doc__ or ""
                self.interface[name] = Toolset._ToolItem(desc, func)
                return func(self, context, tool_input, _scoreboard)
            return inner
        return wrapper

    @staticmethod
    def structurized_tool(tool_name: Optional[str] = None, description: Optional[str] = None, format: Optional[Literal["json", "yaml", "toml"]] = None, **kwargs):
        if format is None:
            format = settings.STRUCTURIZED_TOOL_INPUT_FORMAT
        def wrapper(func: Callable[[...], Any]) -> ToolFunction:
            desc = description or func.__doc__ or ""
            desc += f"\nThe input arguments should be in {format} format."
            @wraps(func)
            def inner(self, tool_input: Any, _scoreboard: Scoreboard) -> str:
                match format:
                    case "json":
                        _kwargs = tool_input
                    case "yaml":
                        import yaml
                        _kwargs = yaml.load(tool_input)
                    case "toml":
                        import toml
                        _kwargs = toml.load(tool_input)
                    case _:
                        raise ValueError(f"Invalid type: {format}")
                _kwargs["scoreboard"] = _scoreboard
                return func(self, **_kwargs)
            return Toolset.tool(description=desc, tool_name=tool_name, **kwargs)(inner)
        return wrapper

def parse_llm_output(output: str) -> Tuple[Optional[str], Optional[str], Optional[str | Dict[str, Any]]]:
    regex = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    match : List[str] = regex.findall(output)
    if len(match) > 1:
        raise MultipleToolCallException()
    if len(match) == 0:
        return None, None, None
    tool_call = match[0]
    try:
        # RWKV MITIGATION

        tool_call = tool_call.replace("'", '"')

        tool_call_json = json.loads(tool_call)
        return tool_call_json["context"], tool_call_json["tool"], tool_call_json.get("args", {})
    except Exception as e:
        print(e)
        raise InvalidToolCallJSONException()
    