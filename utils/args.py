from typing import Callable, Dict, Any
import inspect

from .exceptions import InvalidToolArgsException

def tool_args_guard(func: Callable[[...], Any], args: Dict[str, Any], exclude_self: bool = True):
    """Check if the tool args satisfied function signature.
    
    Automatically removes _context and _scoreboard_manager if not exists.
    """
    sig = inspect.signature(func)
    if exclude_self:
        func_params_list = list(sig.parameters.items())[1:]
        func_params = { k: v for k, v in func_params_list }
    else:
        func_params = sig.parameters
    missing_args = []
    unexpected_args = []
    for name, param in func_params.items():
        if param.default == inspect.Parameter.empty and name not in args:
            # Missing required param
            missing_args.append(name)
    for arg in list(args.keys()):
        # TODO: handle kwargs
        if arg not in sig.parameters:
            if arg.startswith("_"):
                args.pop(arg)
            else:
                unexpected_args.append(arg)
    if len(missing_args) > 0:
        raise InvalidToolArgsException(f"Missing required args: {missing_args}")
    if len(unexpected_args) > 0:
        raise InvalidToolArgsException(f"Unexpected args: {unexpected_args}")