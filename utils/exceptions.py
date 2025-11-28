from . import settings

class ToolCallException(Exception):
    def __init__(self, message: str, penalty: float):
        super().__init__("Tool Call Exception: " + message)
        self.penalty = penalty

class MultipleToolCallException(ToolCallException):
    def __init__(self):
        super().__init__(f"Calling more than 1 tool. At most 1 tool can be called in each step.", settings.INVALID_TOOL_CALL_PENALTY)

class InvalidToolException(ToolCallException):
    def __init__(self, tag: str):
        super().__init__(f"Invalid tool name: `{tag}`.", settings.INVALID_TOOL_CALL_PENALTY)

class InvalidToolCallJSONException(ToolCallException):
    def __init__(self):
        super().__init__('Invalid tool call JSON. Expected format: { "context": "context name", "tool": "tool name", "args": (optional, in json format) }', settings.INVALID_TOOL_CALL_PENALTY)

class ToolNotExistException(ToolCallException):
    def __init__(self, context: str, interface: str):
        super().__init__(f"Interface `{interface}` does not exist in context `{context}`.", settings.INVALID_TOOL_CALL_PENALTY)
        
class InvalidToolInputException(ToolCallException):
    def __init__(self, input: str, expected: str):
        super().__init__(f"Invalid tool input. Expected input: `{expected}`, Received: `{input}`", settings.INVALID_TOOL_CALL_PENALTY)

class ContextNotExistException(ToolCallException):
    def __init__(self, context: str):
        super().__init__(f"Context `{context}` does not exist.", settings.INVALID_TOOL_CALL_PENALTY)
