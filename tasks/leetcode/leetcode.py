from typing import Optional, Dict
import os
import subprocess

from utils.tool import Toolset
from utils.docker import DockerEnvironment
from utils.environment import Environment
from utils.exceptions import ToolCallException
from utils.scoring import DefaultScoreboardManager, ScoreboardManager
from environment.external_tools.shell import DockerShellTool
from environment.internal_tools.self_sft import SelfSFT

LEETCODE_PROMPT = """
You are a helpful assistant that helps the user to solve Leetcode problems.
You are given a problem statement and a code template.
You need to solve the problem and return the solution.
"""

class LeetcodeAPIException(ToolCallException):
    def __init__(self, message: str):
        super().__init__(message, -50)
class LeetcodeAPI:
    def __init__(self, environment: DockerEnvironment):
        self.environment = environment

    def _run_cmd(self, cmd: str):
        result = self.environment.execute(cmd)
        return result["output"], result["returncode"]

    def submit(self, qid: str) -> None:
        output, returncode = self._run_cmd(f"leetgo submit {qid}")
        if returncode != 0:
            raise LeetcodeAPIException(f"Failed to submit: {output}")
        if "Accepted" in output:
            return True, output
        else:
            return False, output

    def pick(self, qid: str) -> None:
        output, returncode = self._run_cmd(f"pwd")
        print("pwd output:", output)
        output, returncode = self._run_cmd(f"ls -la")
        print("ls -la output:", output)
        output, returncode = self._run_cmd(f"leetgo pick {qid}")
        if returncode != 0:
            raise LeetcodeAPIException(f"Failed to pick: {output}")
        output, returncode = self._run_cmd(f"ls -la")
        print("ls -la output:", output)
        return output

class LeetcodeSubmitTool(Toolset):
    def __init__(self, submission_limit: int, leetcode_api: LeetcodeAPI, qid: str):
        super().__init__()
        self.submission_limit = submission_limit
        self.leetcode_api = leetcode_api
        self.qid = qid
        self.submission_attempts = 0

    @Toolset.structurized_tool()
    def submit(self, _scoreboard_manager: ScoreboardManager) -> str:
        """Submit the given code to the Leetcode platform.
        """
        self.submission_attempts += 1
        accepted, result = self.leetcode_api.submit(self.qid)
        if accepted:
            _scoreboard_manager.get_scoreboard().reward(1000, "Submission succeeded.")
        if self.submission_attempts >= self.submission_limit:
            _scoreboard_manager.get_scoreboard().reward(-100, "Submission failed after maximum attempts.")
            self.finish()
        return result["output"]

    @Toolset.structurized_tool(tool_name="finish")
    def finish_task(self) -> None:
        """Finish the task. You should use the tool to end the session only when your solution is accepted.
        """
        self.finish()

class LeetcodeEnvironment(Environment):
    def __init__(self, prompt: str, qid: str, login_credentials: Dict[str, str], tools: Dict[str, Toolset], submission_limit: int = 10, max_steps: int = 100):
        self.submission_limit = submission_limit
        self.environment = DockerEnvironment(image="leetcode-sandbox", cwd="/workspace",
            env=login_credentials)
        self.api = LeetcodeAPI(self.environment)
        self.api.pick(qid)
        leetcode_tool = LeetcodeSubmitTool(submission_limit, self.api, qid)
        super().__init__(tools={
                "leetcode": leetcode_tool,
                "shell-sandbox": DockerShellTool(self.environment),
                **tools,
            }, scoreboard_manager=DefaultScoreboardManager(), prompt={
                "prompt": prompt,
            }, stop_criteria=lambda x: leetcode_tool.finish_flag, max_steps=max_steps)
