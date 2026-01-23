from typing import Dict, Any

from utils.tool import Toolset
from utils.shell import ShellEnvironment

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
        return self.environment.execute(command, cwd=cwd, timeout=timeout)