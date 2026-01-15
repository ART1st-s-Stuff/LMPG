from typing import Optional, Dict
import os
import subprocess

from utils.tool import Toolset
from utils.docker import DockerEnvironment
from utils.environment import Environment
from utils.scoring import DefaultScoreboardManager
from environment.external_tools.shell import DockerShellTool
from environment.internal_tools.self_sft import SelfSFT

LEETCODE_PROMPT = """
You are a helpful assistant that helps the user to solve Leetcode problems.
You are given a problem statement and a code template.
You need to solve the problem and return the solution.
"""

class LeetcodeAPIException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
class LeetcodeAPI:
    def __init__(self, login_credentials: Dict[str, str]):
        self.env = os.environ.copy()
        self.env.update(login_credentials)

    def _run_cmd(self, cmd: str) -> None:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False, env=self.env)
        return result.stdout, result.returncode

    def submit(self, qid: str) -> None:
        output, returncode = self._run_cmd(f"leetgo submit {qid}")
        if returncode != 0:
            raise LeetcodeAPIException(f"Failed to submit: {output}")
        if "Accepted" in output:
            return True, output
        else:
            return False, output

    def pick(self, qid: str) -> None:
        output, returncode = self._run_cmd(f"leetgo pick {qid}")
        if returncode != 0:
            raise LeetcodeAPIException(f"Failed to pick: {output}")
        return output

class LeetcodeSubmitTool(Toolset):
    def __init__(self, submission_limit: int, environment: DockerEnvironment, leetcode_api: LeetcodeAPI, qid: str):
        super().__init__()
        self.environment = environment
        self.submission_limit = submission_limit
        self.leetcode_api = leetcode_api
        self.qid = qid

    @Toolset.structurized_tool()
    def submit(self, q: str) -> None:
        """Submit the given code to the Leetcode platform.
        """
        accepted, result = self.leetcode_api.submit(self.qid)
        if accepted:
            self.finish()
        return result

class LeetcodeEnvironment(Environment):
    def __init__(self, prompt: str, qid: str, login_credentials: Dict[str, str], submission_limit: int = 10):
        self.submission_limit = submission_limit
        self.environment = DockerEnvironment(image="leetcode-sandbox", cwd="/leetcode/cpp")
        self.api = LeetcodeAPI(login_credentials)
        self.api.pick(qid)
        super().__init__(tools={
                "leetcode": LeetcodeSubmitTool(submission_limit, self.environment, self.api, qid),
                "shell-sandbox": DockerShellTool(self.environment),
                "self-sft": SelfSFT(),
            }, scoreboard_manager=DefaultScoreboardManager(), prompt={
                "task": prompt,
            }, stop_criteria=None, max_steps=submission_limit)
