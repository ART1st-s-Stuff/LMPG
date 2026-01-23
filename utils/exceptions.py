from . import settings

class ToolCallException(Exception):
    def __init__(self, message: str, penalty: float = settings.INVALID_TOOL_CALL_PENALTY):
        super().__init__("Tool Call Exception: " + message)
        self.penalty = penalty

class MultipleToolCallException(ToolCallException):
    def __init__(self):
        super().__init__(f"Calling more than 1 tool. At most 1 tool can be called in each step.", settings.INVALID_TOOL_CALL_PENALTY)

class InvalidToolCallJSONException(ToolCallException):
    def __init__(self):
        super().__init__('Invalid tool call JSON. Expected format: { "tool_set": "<toolset name>", "tool_name": "<tool name>", "args": (in json format, can be empty) }', settings.INVALID_TOOL_CALL_PENALTY)

class ToolNotExistException(ToolCallException):
    def __init__(self, tool_set: str, tool_name: str):
        super().__init__(f"tool_name `{tool_name}` does not exist in tool_set `{tool_set}`.", settings.INVALID_TOOL_CALL_PENALTY)
        
class InvalidToolArgsException(ToolCallException):
    def __init__(self, text: str):
        super().__init__(f"Invalid tool args: {text}", settings.INVALID_TOOL_CALL_PENALTY)

class ToolsetNotExistException(ToolCallException):
    def __init__(self, tool_set: str):
        super().__init__(f"Tool_set `{tool_set}` does not exist.", settings.INVALID_TOOL_CALL_PENALTY)