from typing import Literal, Callable, Sequence, Dict, Any
from abc import ABC, abstractmethod
from textwrap import dedent

from . import settings
from .exceptions import ToolNotExistException
from .args import tool_args_guard

class TextWindow(ABC):
    def __init__(self, window_id: str, interface_prefix: str, volatile: bool = False):
        self.interface_prefix = interface_prefix
        self.window_id = window_id
        self.volatile = volatile

    def invoke(self, tool: str, tool_input: Dict[str, Any]) -> str:
        if tool in self.interface:
            func = self.interface[tool.lower()]
            # Do not exclude self, since it is already excluded
            tool_args_guard(func, tool_input, exclude_self=False)
            return func(**tool_input)
        else:
            raise ToolNotExistException(self.window_name, tool)

    @property
    @abstractmethod
    def interface(self):
        ...

    @abstractmethod
    def read(self) -> str:
        ...

    @property
    @abstractmethod
    def window_name(self) -> str:
        ...

    @property
    def _window_name(self) -> str:
        return f"{self.interface_prefix}-{self.window_id}"

    @classmethod
    @abstractmethod
    def hint(cls) -> str:
        ...
    
    # def embed_roles(self, tokens: torch.Tensor) -> torch.Tensor:
    #     dim_role = torch.zeros(tokens.size(0))
    #     return torch.stack([tokens, dim_role], dim=-1)

class SegmentTextWindow(TextWindow):
    """A list of paragraphs, split into segments."""
    SEGMENT_LENGTH = settings.TEXT_WINDOW_SEGMENT_LENGTH

    def __init__(self, tokenizer, text: Any, window_id: str, interface_prefix: str, volatile: bool = False):
        super().__init__(window_id, interface_prefix, volatile)
        self.tokenizer = tokenizer
        if not isinstance(text, str):
            text = str(text)
        self.segments = []
        tokens = self.tokenizer.encode(text)
        for i in range(0, len(tokens), self.SEGMENT_LENGTH):
            self.segments.append(self.tokenizer.decode(tokens[i:i + self.SEGMENT_LENGTH]))
        self.current_segment = 1

    @property
    def interface(self):
        return {
            "read": self.read,
            "go_to_segment": self.go_to_segment
        }
    
    def read(self) -> str:
        if self.current_segment >= len(self.segments):
            return f"<[{self.window_name}]>Max segment number: {len(self.segments)}<[/{self.window_name}]>"
        ret = f"<[WINDOW{self.window_name}][PARAGRAPH {self.current_segment} OF {len(self.segments)}]>" + self.segments[self.current_segment - 1] + "<[END_OF_PARAGRAPH]>"
        return ret

    def go_to_segment(self, segment_number: int) -> str:
        self.current_segment = segment_number
        return self.read()

    @classmethod
    def hint(cls) -> str:
        return dedent(
            """
            Text segment window: The long text is split into segments.

            read: Read the current segment.
            go_to_segment: Go to a specific segment and read. Args: segment_number, int: The number of the segment to go to.
            """
        )

    @property
    def window_name(self) -> str:
        return f"text-{self._window_name}"

class FileTextWindow(TextWindow):
    LINES_IN_A_WINDOW = 40

    def __init__(self, text: str, window_id: str, interface_prefix: str, volatile: bool = False):
        super().__init__(window_id, interface_prefix, volatile)
        lines = text.split('\n')
        self.lines = [f"<[LINE-{i}]>{line}" for i, line in enumerate(lines)]
        self.current_line = 1

    @property
    def interface(self):
        return {
            "read": self.read,
            "go_to_line": self.go_to_line
        }

    def read(self) -> str:
        start_line = self.current_line - 1
        end_line = start_line + self.LINES_IN_A_WINDOW
        lines = self.lines[start_line:end_line]
        ret = '\n'.join(lines)
        if end_line < len(self.lines):
            ret = ret + "<[END_OF_TEXT]>"
        return f"<[{self.window_name}]>{ret}<[/{self.window_name}]>"

    def go_to_line(self, line_number: int) -> str:
        self.current_line = line_number
        return self.read()

    @property
    def window_name(self) -> str:
        return f"file-{self._window_name}"

    @classmethod
    def hint(cls) -> str:
        return dedent(
            f"""
            File window: View a file. A window contains {cls.LINES_IN_A_WINDOW} lines.

            read: Read the current window.
            go_to_line: Go to a specific line and read. Args: line_number, int: The number of the line to go to.
            """
        )
    
def text_window(text: str | Sequence[str], window_id: str, interface_prefix: str, window_type: Literal['segment', 'file'], volatile: bool = False) -> TextWindow:
    if window_type == 'segment':
        return SegmentTextWindow(settings.TOKENIZER, text, window_id, interface_prefix, volatile)
    elif window_type == 'file':
        return FileTextWindow(text, window_id, interface_prefix, volatile)
    else:
        raise ValueError(f"Invalid window type: {window_type}")