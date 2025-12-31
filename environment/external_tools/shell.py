from abc import abstractmethod, ABC
from typing import Dict, Any
import subprocess
import os

from utils.tool import Toolset
from utils.docker import DockerEnvironment


class ShellTool(Toolset, ABC):
    @abstractmethod
    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the shell.

        Args:
            command, str: The command to execute.
            cwd, str: The working directory to execute the command in. Set to "" for default working directory.
            timeout, int: The timeout in seconds for the command to execute.
        """
        ...

class LocalShellTool(ShellTool):
    def __init__(self, cwd: str):
        super().__init__()
        self.cwd = cwd

    @Toolset.structurized_tool()
    def execute(self, command: str, cwd: str = "", timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the shell.

        Args:
            command, str: The command to execute.
            cwd, str: The working directory to execute the command in. Set to "" for default working directory.
            timeout, int: The timeout in seconds for the command to execute.
        """
        cwd = cwd or self.cwd
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            cwd=cwd,
            env=os.environ | self.config.env,
            timeout=timeout or self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

class DockerShellTool(ShellTool):
    def __init__(self, environment: DockerEnvironment):
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