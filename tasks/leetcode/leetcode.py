from typing import Optional
import os
import subprocess

from utils.tool import Toolset
from utils.docker import DockerEnvironment

class LeetcodeAPIException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
class LeetcodeAPI:
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        username = username or os.getenv("LEETCODE_USERNAME")
        password = password or os.getenv("LEETCODE_PASSWORD")
        assert username is not None and password is not None, "Username and password for Leetcode API are required"
        assert self._run_cmd("leetcode")[1] == 0, "Failed to run leetcode command"
        self.login(username, password)

    def _run_cmd(self, cmd: str) -> None:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.returncode

    def login(self, username: str, password: str) -> None:
        process = subprocess.Popen(
            "leetcode login",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        output, error = process.communicate(input=username)
        output, error = process.communicate(input=password)
        if process.returncode != 0:
            raise LeetcodeAPIException(f"Failed to login: {error}")

    def submit(self, src: str) -> None:
        output, returncode = self._run_cmd(f"leetcode submit {src}")
        if returncode != 0:
            raise LeetcodeAPIException(f"Failed to submit: {output}")
        if "Accepted" in output:
            return True, output
        else:
            return False, output

class LeetcodeSubmitTool(Toolset):
    def __init__(self, submission_limit: int, environment: DockerEnvironment, leetcode_api: LeetcodeAPI):
        super().__init__()
        self.environment = environment
        self.submission_limit = submission_limit
        self.leetcode_api = leetcode_api

    @Toolset.structurized_tool()
    def submit(self, src: str) -> None:
        """Submit the given code to the Leetcode platform.

        Args:
            src, str: The source code file to submit.
        """
        accepted, result = self.leetcode_api.submit(src)
        if accepted:
            self.finish()
        return result

class LeetcodeEnvironment(Environment):
    def __init__(self, submission_limit: int):
        super().__init__()
        self.submission_limit = submission_limit
        self.environment = DockerEnvironment(image="leetcode-submit", cwd="skygragon/leetcode-cli")

    #TODO