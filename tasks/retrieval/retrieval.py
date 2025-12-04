from textwrap import dedent
import os
import json
from typing import Optional
import re

from utils.tool import Toolset
from utils.environment import Environment
from utils.scoring import ScoreboardManager, DefaultScoreboardManager
from utils.agent import SFTAgent

import utils.settings as settings

PROMPT = dedent(
    """
    Task:
    Answer the question {question} according to the document in the "document" window.
    Use output tool to output the result. Your output should be a straight forward answer:

    Example question:
    Micheal de Santa accidentally destroyed whose house?
    Example answer:
    Martin Madrazo

    Hints:
    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following parameters in the tool call:
    - View a window: <tool>{{ "context": "window name", "tool": "read" }}</tool>
    - Go to a specific segment: <tool>{{ "context": "window name", "tool": "goto", "args": {{ "segment_number": int }} }}</tool>

    You have access to the following windows:
    - prompt: The prompt of the task, this window.
    - document: The document of the task.

    You have access to the following tool:
    - output: Output the result. 
      - context "output" and tool "output"
      - args: {{ "content": str }}
    """
)

DATASET_FILE = os.path.join(os.path.dirname(__file__), "single_needle_dataset.json")
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    DATASET = json.load(f)

class Output(Toolset):
    def __init__(self, gt_label: str):
        super().__init__()
        self.result = None
        self.gt_label = gt_label

    @Toolset.structurized_tool()
    def output(self, content: str, _scoreboard_manager: ScoreboardManager):
        """Use this tool to output the result."""
        print(f"Agent: {content}")
        self.result = content
        if self.result == self.gt_label:
            _scoreboard_manager.get_scoreboard().reward(100, "Correct answer.")
        else:
            _scoreboard_manager.get_scoreboard().reward(0, "Incorrect answer.")

def build_one_task(prompt: str, text: str, label: str, tokenizer):
    token_length = tokenizer([text], return_tensors="pt").input_ids.shape[1]
    output = Output(label)
    return Environment(
        tools={
            "output": output
        },
        scoreboard_manager=DefaultScoreboardManager(),
        prompt={
            "prompt": PROMPT.format(question=prompt),
            "document": text,
        },
        max_steps=2 * token_length // settings.TEXT_WINDOW_SEGMENT_LENGTH + 20,
        stop_criteria=lambda _: output.result is not None
    )

def build_tasks(num: int, TOKENIZER):
    data = DATASET[:num]
    tasks = [build_one_task(prompt, text, label, TOKENIZER) for prompt, text, label in data]
    return tasks

def long_thinking_penalty_hook(instance, model_output: str, context: Optional[str], tool: Optional[str], tool_input: Optional[str]) -> Optional[str]:
    token_length = instance.tokenizer([model_output], return_tensors="pt").input_ids.shape[1]
    if token_length > 400:
        instance.scoreboard_manager.get_scoreboard().reward(-2.0, "Thinking process length exceeds 400 tokens.")