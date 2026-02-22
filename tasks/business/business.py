from typing import Optional, Dict, Literal, List, Tuple
import os
import subprocess
import shutil
import json
import random
import re

from utils.tool import Toolset
from utils.shell import ShellEnvironment, DockerShellEnvironment, LocalShellEnvironment
from utils.environment import Environment
from utils.exceptions import ToolCallException
from utils.scoring import DefaultScoreboardManager, ScoreboardManager
from environment.external_tools.shell import ShellTool
from environment.internal_tools.self_sft import SelfSFT
from utils.text import text_window

LEETCODE_PROMPT = """
You are a helpful assistant that helps the user to solve Leetcode problems.
You are given a problem statement and a code template.
You need to solve the problem and return the solution.
"""

class BusinessDocumentDB:
    def __init__(self, document_path: str):
        self.document_path = document_path
        gt_file = os.path.join(document_path, "ground_truth.json")
        with open(gt_file, "r", encoding="utf-8") as f:
            self.gt = json.load(f)
        self.valid_samples = { x: y for x, y in self.gt.items() if y != "None" }
        self.empty_samples = { x: y for x, y in self.gt.items() if y == "None" }

    def get_data(self, valid_train_samples: int, empty_train_samples: int, rng_seed: int = 42) -> List[Tuple[str, str]]:
        rng = random.Random(rng_seed)
        valid_train_pairs = rng.sample(list(self.valid_samples.items()), valid_train_samples)
        empty_train_pairs = rng.sample(list(self.empty_samples.items()), empty_train_samples)
        # The rest are test data.
        valid_test_pairs = set(self.valid_samples.items()) - set(valid_train_pairs)
        empty_test_pairs = set(self.empty_samples.items()) - set(empty_train_pairs)
        test_pairs = list(set(valid_test_pairs) | set(empty_test_pairs))
        # Shuffle
        rng.shuffle(test_pairs)
        return {
            "train_valid": valid_train_pairs,
            "train_empty": empty_train_pairs,
            "test": test_pairs,
        }

class AnswerTool(Toolset):
    def __init__(self, db: BusinessDocumentDB, *, valid_train_samples: int, empty_train_samples: int):
        super().__init__()
        ds = db.get_data(valid_train_samples=valid_train_samples, empty_train_samples=empty_train_samples)
        self.test_ds = ds["test"]
        self.current_test_index = -1
        self.answered = True
        self.correct = 0

    @Toolset.structurized_tool()
    def next_document(self, _scoreboard_manager: ScoreboardManager) -> str:
        """Get the next document xml file name.
        """
        self.current_test_index += 1
        if self.answered == False:
            _scoreboard_manager.get_scoreboard().reward(-100, "You should submit the answer to the previous document first.")
            return
        document_id, _ = self.test_ds[self.current_test_index]
        self.answered = False
        return document_id

    @Toolset.structurized_tool()
    def answer(self, answer: str, _scoreboard_manager: ScoreboardManager) -> str:
        """Answer the question based on the given document (in window "document").
        """
        if self.current_test_index == -1:
            _scoreboard_manager.get_scoreboard().reward(-100, "Get a document file first.")
            return
        if self.answered:
            _scoreboard_manager.get_scoreboard().reward(-100, "You can only submit one answer per document.")
            return
        self.answered = True
        _, gt = self.test_ds[self.current_test_index]
        if gt == answer.strip():
            _scoreboard_manager.get_scoreboard().reward(100, "Answer is correct.")
            self.correct += 1
        else:
            _scoreboard_manager.get_scoreboard().reward(0, "Answer is incorrect.")
        if self.current_test_index == len(self.test_ds) - 1:
            print("Correct: ", self.correct, "/", len(self.test_ds))
            self.finish()

class TrainingAnswerTool(Toolset):
    def __init__(self, db: BusinessDocumentDB, environment: BusinessDocumentEnvironment, *, valid_train_samples: int, empty_train_samples: int):
        super().__init__()
        ds = db.get_data(valid_train_samples=valid_train_samples, empty_train_samples=empty_train_samples)
        self.train_ds = ds["train_valid"] + ds["train_empty"]
        random.Random(42).shuffle(self.train_ds)
        self.current_test_index = -1
        self.answered = True
        self.environment = environment

    @Toolset.structurized_tool()
    def next_document(self, _scoreboard_manager: ScoreboardManager) -> str:
        """Get the next document xml file name.
        """
        self.current_test_index += 1
        if  self.current_test_index >= len(self.train_ds):
            self.finish()
            return
        document_id, _ = self.train_ds[self.current_test_index]
        self.environment.current_document = document_id
        if self.answered == False:
            _scoreboard_manager.get_scoreboard().reward(-100, "You should submit the answer to the previous document first.")
            return
        self.answered = False
        if self.current_test_index == len(self.train_ds) - 1:
            return "This is the last document. Document xml file name: " + document_id
        return "Document xml file name: " + document_id

    @Toolset.structurized_tool()
    def answer(self, answer: str, _scoreboard_manager: ScoreboardManager) -> str:
        """Answer the question based on the given document (in window "document").
        """
        if self.current_test_index == -1:
            _scoreboard_manager.get_scoreboard().reward(-100, "Get a document file first.")
            return
        if self.answered:
            _scoreboard_manager.get_scoreboard().reward(-100, "You can only submit one answer per document.")
            return
        self.answered = True
        _, gt = self.train_ds[self.current_test_index]
        if gt == answer.strip():
            _scoreboard_manager.get_scoreboard().reward(100, "Answer is correct.")
            return "The answer is correct."
        else:
            if answer != "":
                _scoreboard_manager.get_scoreboard().reward(0, "Answer is incorrect.")
                return "Your answer is incorrect. The correct answer is: " + gt
            else:
                return "The correct answer is: " + gt

    @Toolset.structurized_tool(tool_name="finish")
    def finish_tool(self, _scoreboard_manager: ScoreboardManager) -> str:
        """Finish the training session."""
        self.finish()

class ShellToolWithReward(Toolset):
    def __init__(self, environment: ShellEnvironment, task_environment: BusinessDocumentEnvironment):
        super().__init__()
        self.environment = environment
        self.task_environment = task_environment
        self.granted_files = set()

    @Toolset.structurized_tool()
    def execute(self, _scoreboard_manager: ScoreboardManager, command: str, cwd: str = "", timeout: int | None = None) -> Dict[str, Any]:
        """Execute a command in the shell.

        Args:
            command, str: The command to execute.
            cwd, str: The working directory to execute the command in. Set to "" for default working directory.
            timeout, int: The timeout in seconds for the command to execute.
        """
        try:
            result = self.environment.execute(command, cwd=cwd, timeout=timeout)
            if result["returncode"] == 0:
                if self.task_environment.current_document not in self.granted_files and \
                        re.match(f"sed -n '\d+,\d+p' {self.task_environment.current_document}", command):
                    _scoreboard_manager.get_scoreboard().reward(5, "Successfully read the document.")
                    self.granted_files.add(self.task_environment.current_document)
            else:
                _scoreboard_manager.get_scoreboard().reward(-5, "Command executed with errors.")
            return text_window(
                text=result,
                window_id="output",
                interface_prefix="shell",
                window_type="segment"
            )
        except Exception as e:
            raise ToolCallException(f"Failed to execute command: {command} due to {e}")

class BusinessDocumentEnvironment(Environment):
    def __init__(self, prompt: str, db: BusinessDocumentDB, tools: Dict[str, Toolset], *,
            max_steps: int = 100, training: bool = False, valid_train_samples: int = 100, empty_train_samples: int = 100):
        self.db = db
        self.shell_environment = LocalShellEnvironment(cwd=db.document_path)
        answer_tool = AnswerTool(self.db, valid_train_samples=valid_train_samples, empty_train_samples=empty_train_samples) if not training else TrainingAnswerTool(self.db, valid_train_samples=valid_train_samples, empty_train_samples=empty_train_samples)
        self.current_document : Optional[str] = None
        super().__init__(tools={
                "answer": answer_tool,
                "shell": ShellTool(self.shell_environment),
                **tools,
            }, scoreboard_manager=DefaultScoreboardManager(), prompt={
                "prompt": prompt,
            }, stop_criteria=lambda x: answer_tool.finish_flag, max_steps=max_steps)