from typing import Optional, Dict, Literal, List, Tuple
import os
import subprocess
import shutil
import json
import random

from utils.tool import Toolset
from utils.shell import ShellEnvironment, DockerShellEnvironment, LocalShellEnvironment
from utils.environment import Environment
from utils.exceptions import ToolCallException
from utils.scoring import DefaultScoreboardManager, ScoreboardManager
from environment.external_tools.shell import ShellTool
from environment.internal_tools.self_sft import SelfSFT

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
        self.valid_samples = { x: y if y is not None else "None" for x, y in self.gt.items() if y is not None }
        self.empty_samples = { x: y if y is not None else "None" for x, y in self.gt.items() if y is None }

    def get_data(self, valid_train_samples: int, empty_train_samples: int, rng_seed: int = 42) -> List[Tuple[str, str]]:
        rng = random.Random(rng_seed)
        valid_train_pairs = rng.sample(list(self.valid_samples.items()), valid_train_samples)
        empty_train_pairs = rng.sample(list(self.empty_samples.items()), empty_train_samples)
        # The rest are test data.
        valid_test_pairs = list(self.valid_samples.items()) - valid_train_pairs
        empty_test_pairs = list(self.empty_samples.items()) - empty_train_pairs
        test_pairs = valid_test_pairs + empty_test_pairs
        # Shuffle
        rng.shuffle(test_pairs)
        return {
            "train_valid": valid_train_pairs,
            "train_empty": empty_train_pairs,
            "test": test_pairs,
        }

class AnswerTool(Toolset):
    def __init__(self, db: BusinessDocumentDB):
        super().__init__()
        ds = db.get_data()
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
    def __init__(self, db: BusinessDocumentDB, *, valid_train_samples: int, empty_train_samples: int):
        super().__init__()
        ds = db.get_data(valid_train_samples=valid_train_samples, empty_train_samples=empty_train_samples)
        self.train_ds = ds["train_valid"] + ds["train_empty"]
        random.Random(42).shuffle(self.train_ds)
        self.current_test_index = -1
        self.answered = True

    @Toolset.structurized_tool()
    def next_document(self, _scoreboard_manager: ScoreboardManager) -> str:
        """Get the next document xml file name.
        """
        self.current_test_index += 1
        if  self.current_test_index >= len(self.train_ds):
            self.finish()
            return
        document_id, _ = self.train_ds[self.current_test_index]
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

class BusinessDocumentEnvironment(Environment):
    def __init__(self, prompt: str, db: BusinessDocumentDB, tools: Dict[str, Toolset], *,
            max_steps: int = 100, training: bool = False, valid_train_samples: int = 100, empty_train_samples: int = 100):
        self.db = db
        self.shell_environment = LocalShellEnvironment(cwd=db.document_path)
        answer_tool = AnswerTool(self.db) if not training else TrainingAnswerTool(self.db, valid_train_samples=valid_train_samples, empty_train_samples=empty_train_samples)
        super().__init__(tools={
                "answer": answer_tool,
                "shell": ShellTool(self.shell_environment),
                **tools,
            }, scoreboard_manager=DefaultScoreboardManager(), prompt={
                "prompt": prompt,
            }, stop_criteria=lambda x: answer_tool.finish_flag, max_steps=max_steps)
