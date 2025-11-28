from textwrap import dedent
import os
import json
from typing import Optional
import re

from utils.tool import Toolset
from utils.environment import ManualStoppingEnvironment
from utils.scoring import DefaultScoreboardManager, Scoreboard
from utils.agent import SFTAgent

import utils.settings as settings

HINT_TRAIN = dedent(
    """
    You are an agent in the training process. Your ultimate goal is to effectively
    complete various types of tasks. Now, you are in training stage 1: learning to
    use the tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call. The tool call must be in the following
    format:
        <tool_call>{ "context": "context name", "tool": "tool name", "args": (optional, in json format) }</tool_call>
    You may also choose not to use any tools. All your output other than the tool call
    will be coonsidered as your thinking process.

    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following operations:
    - View a window: <tool_call>{ "context": "window name", "tool": "read" }</tool_call>
    - Go to a specific segment: <tool_call>{ "context": "window name", "tool": "goto", "args": { "segment_number": int } }</tool_call>

    In this stage, you have access to the following windows:
    - TEXT-default-hint: This hint.
    - TEXT-default-task: The task description.
    - TEXT-default-document: The document.
    - window_list: List of all the windows.
    
    You also have access to some tools:
    - output: Output the result.
    - end: End the task.
    - self_sft: Self-supervised training. You can use this tool to train yourself.
                Through SFT, you can memorize some useful information. You may summarize
                some useful information after finishing the task, and call this tool
                to memorize it.

    Do not output too much in a single turn. The output limit is ~500 tokens. You may
    receive a penalty if you do so.

    It is suggested to first open the "TEXT-default-task" window to read the task:
        <tool_call>{ "context": "TEXT-default-task", "tool": "read" }</tool_call>

    Try to achieve higher score in the task.
    """
)

TASK_PROMPT = dedent(
    """
    Answer the question based on the given document (in window "document").

    Question: {question}
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
    def output(self, content: str, _scoreboard: Scoreboard):
        """Use this tool to output the result."""
        print(f"Agent: {content}")
        self.result = content
        if self.result == self.gt_label:
            _scoreboard.reward(100, "Correct answer.")
        else:
            _scoreboard.reward(0, "Incorrect answer.")

def build_one_task(prompt: str, text: str, label: str):
    return ManualStoppingEnvironment(
        tools={
            "output": Output(label)
        },
        scoreboard_manager=DefaultScoreboardManager(),
        prompt={
            "hint": HINT_TRAIN,
            "task": TASK_PROMPT.format(question=prompt),
            "document": text,
        },
        max_steps=500
    )

def build_tasks(num: int):
    data = DATASET[:num]
    tasks = [build_one_task(prompt, text, label) for prompt, text, label in data]
    return tasks

def long_thinking_penalty_hook(instance: SFTAgent, model_output: str, context: Optional[str], tool: Optional[str], tool_input: Optional[str]) -> Optional[str]:
    regex = re.compile(r'<[(.*?)]>(.*)<[/(.*?)]>')
    output = re.sub(regex, r'', model_output)
    token_length = instance.tokenizer([output], return_tensors="pt").input_ids.shape[1]
    if token_length > 400:
        instance.scoreboard_manager.get_scoreboard().reward(-2.0, "Thinking process length exceeds 400 tokens.")