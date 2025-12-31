from abc import ABC
import re
from typing import List, Tuple, Callable, Literal, Any, TypeVar, Optional, Dict, NamedTuple, Generic
from functools import wraps
import json
from collections import ChainMap

from . import settings
from .scoring import ScoreboardManager
from .exceptions import ToolNotExistException, MultipleToolCallException, InvalidToolException, InvalidToolCallJSONException
from .args import tool_args_guard


T = TypeVar('T', bound='Toolset')
ToolFunction = Callable[[T, str, Any, ScoreboardManager], str]

class __ToolsetMeta(type(ABC)):
    def __new__(mcs, name, bases, attrs, /, **kwargs):
        cls = super().__new__(mcs, name, bases, attrs, **kwargs)
        cls_interface : Dict[str, ToolFunction] = {}
        for name, attr in attrs.items():
            if hasattr(attr, "__is_tool__"):
                cls_interface[attr.__tool_name__] = attr
        cls_mro_interface = ChainMap(cls_interface, *[ pcls.__cls_interface__ for pcls in cls.__mro__ if hasattr(pcls, "__cls_interface__") ])
        cls.__cls_interface__ = cls_mro_interface
        return cls

class ToolsetInterfaceIterator:
    def __init__(self, interface_funcs):
        self.iter = iter(interface_funcs.items())

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, str, ToolFunction]:
        name, func = self.iter.__next__()
        return name, func.__tool_desc__, func

class Toolset(metaclass=__ToolsetMeta):
    finish_flag: bool = False

    def __init__(self):
        self.__interface__ = dict(self.__class__.__cls_interface__)

    @property
    def interface(self):
        return ToolsetInterfaceIterator(self.__interface__)

    def invoke(self, tool_name: str, tool_input: Any, tool_set: str, scoreboard_manager: ScoreboardManager) -> str:
        key = tool_name.lower()
        if key not in self.__interface__:
            raise ToolNotExistException(tool_set, tool_name)
        func = self.__interface__[key]
        return func(self, tool_input, _tool_set=tool_set, _scoreboard_manager=scoreboard_manager)

    @staticmethod
    def tool(tool_name: Optional[str] = None, description: Optional[str] = None) -> Callable[[ToolFunction], ToolFunction]:
        def wrapper(func: ToolFunction):
            func.__is_tool__ = True
            func.__tool_name__ = (tool_name or func.__name__).lower()
            func.__tool_desc__ = description or func.__doc__ or ""
            print(f"Adding interface {tool_name or func.__name__}")
            # @wraps(func)
            # def inner(self, context: str, tool_input: str, _scoreboard: Scoreboard) -> str:
            #     return func(self, context, tool_input, _scoreboard)
            return func
        return wrapper

    @staticmethod
    def structurized_tool(tool_name: Optional[str] = None, description: Optional[str] = None, format: Optional[Literal["json", "yaml", "toml"]] = None, **kwargs):
        if format is None:
            format = settings.STRUCTURIZED_TOOL_INPUT_FORMAT
        def wrapper(func: Callable[[...], Any]) -> ToolFunction:
            desc = description or func.__doc__ or ""
            desc += f"\nThe input arguments should be in {format} format."
            @wraps(func)
            def inner(self, tool_input: Any, _tool_set: str, _scoreboard_manager: ScoreboardManager) -> str:
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
                _kwargs["_tool_set"] = _tool_set
                _kwargs["_scoreboard_manager"] = _scoreboard_manager
                tool_args_guard(func, tool_input)
                return func(self, **_kwargs)
            return Toolset.tool(description=desc, tool_name=tool_name, **kwargs)(inner)
        return wrapper

    def finish(self):
        self.finish_flag = True

def parse_llm_output(output: str) -> Tuple[Optional[str], Optional[str], Optional[str | Dict[str, Any]]]:
    regex = re.compile(r'<tool>(.*?)</tool>', re.DOTALL)
    match : List[str] = regex.findall(output)
    if len(match) > 1:
        raise MultipleToolCallException()
    if len(match) == 0:
        return None, None, None
    tool_call = match[0]
    try:
        tool_call_json = json.loads(tool_call)
        assert isinstance(tool_call_json["tool_set"], str)
        assert isinstance(tool_call_json["tool_name"], str)
        assert isinstance(tool_call_json.get("args", {}), dict)
        return tool_call_json["tool_set"], tool_call_json["tool_name"], tool_call_json.get("args", {})
    except Exception as e:
        print(e)
        raise InvalidToolCallJSONException()
    