from typing import Dict, Any

from utils.tool import Toolset
from utils.shell import ShellEnvironment
from utils.exceptions import ToolCallException
from utils.text import text_window

class ShellTool(Toolset):
    def __init__(self, environment: ShellEnvironment):
        super().__init__()
        self.environment = environment

    @Toolset.structurized_tool()
    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the shell.

        Args:
            command, str: The command to execute.
            cwd, str: The working directory to execute the command in. Set to "" for default working directory.
            timeout, int: The timeout in seconds for the command to execute.
        """
        try:
            return text_window(
                text=self.environment.execute(command, cwd=cwd, timeout=timeout),
                window_id="output",
                interface_prefix="shell",
                window_type="segment"
            )
        except Exception as e:
            raise ToolCallException(f"Failed to execute command: {command} due to {e}")