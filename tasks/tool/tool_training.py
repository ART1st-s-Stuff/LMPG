from textwrap import dedent
import os
import json
from typing import Dict
import random
import string

from utils.tool import Toolset
from utils.environment import Environment
from utils.scoring import DefaultScoreboardManager, ScoreboardManager

HINT_TRAIN = dedent(
    """
    You are an agent in the training process. Your are learning to use tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call. You must wrap the tool call within <tool></tool>
    tags, and follow the following specification:
    <tool>{ "context": "context name", "tool": "tool name", "args": (optional, in json format) }</tool>
    The "context" and "tool" fields are case sensitive.

    Example output 1:
    ```
    I have finished step 1, but I have forgotten the next step to do. I need to open the prompt
    again to decide my next action.
    <tool>{ "context": "text-default-prompt", "tool": "read" }</tool>
    ```
    Example output 2:
    ```
    I have found the answer 29380. Now I output the answer.
    <tool>{ "context": "interact", "tool": "output", "args": { "content" : 29380 } }</tool>
    ```
    Example output 3:
    ```
    We should run the program again and expect a successful output.
    But the log suggests we cannot connect to the API. However, we have previously set the
    proxy and curl test to the API is fine. Perhaps it is the due to library requires
    explicit proxy settings.
    I should try to figure out if the proxy settings is correct in the next step.
    ```

    You may also choose not to use any tools. All your output other than the tool call
    will be coonsidered as your thinking process.

    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following parameters in the tool call:
    - View a window: <tool>{ "context": "window name", "tool": "read" }</tool>
    - Go to a specific segment: <tool>{ "context": "window name", "tool": "goto", "args": { "segment_number": int } }</tool>

    In this stage, you have access to the following windows:
    - text-default-prompt: This prompt.
    - text-default-<window_name>: The window you should open in step 1. It tells you what to do next.

    Your task:
    Step 1. Open the "text-default-step1" window, and follow the instruction there.
    """
)

HINT_RWKV_OFFICIAL = dedent(
    """
    You are an agent in the training process. Your are learning to use tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call.

    You may also choose not to use any tools. All your output other than the tool call
    will be coonsidered as your thinking process.

    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following parameters in the tool call:
    - View a window: { "context": "window name", "tool": "read" }
    - Go to a specific segment: { "context": "window name", "tool": "goto", "args": { "segment_number": int } }

    In this stage, you have access to the following windows:
    - text-default-prompt: This prompt.
    - window_list: List of all the windows.
    
    You also have access to another tools:
    - end: End the task.

    Your task:
    Step 1. Open the "<window_name>" window.
    Step 2. Use the end tool to end the task.

    Tool call specification:
    {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "tool",
                    "description": "Execute a tool call",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "string",
                                "description": "The context of the tool call"
                            },
                            "tool": {
                                "type": "string",
                                "description": "The name of the tool to execute"
                            },
                            "args": {
                                "type": "object",
                                "description": "The arguments of the tool call"
                            },
                        },
                        "required": ["context", "tool"]
                    }
                }
            }
        ]
    }
    """
)

HINT_STEP1 = dedent(
    """
    You finished step 1 of the task.

    You should output the result "<result_str>"

    Continue to step 2: Use the "output" tool to output the result and end the task. It has context "output" and tool "output", its args schema is { "content" : "str output" }.
    """
)

TASK_PROMPT = dedent(
    """
    Answer the question based on the given document (in window "document").

    Question: {question}
    """
)


class Output(Toolset):
    def __init__(self, gt):
        super().__init__()
        self.stopped = False
        self.gt = gt
    
    @Toolset.structurized_tool()
    def output(self, content: str, _scoreboard_manager: ScoreboardManager):
        """Use this tool output the result and to end the task."""
        self.stopped = True
        if content == self.gt:
            _scoreboard_manager.get_scoreboard().reward(1000)

class ManualStoppingEnvironment(Environment):
    def __init__(self, tools: Dict[str, Toolset], scoreboard_manager: ScoreboardManager, prompt: Dict[str, str], max_steps: int, gt_label: str):
        output_tool = Output(gt_label)
        super().__init__({ **tools, "output": output_tool }, scoreboard_manager, prompt, lambda x: output_tool.stopped, max_steps)

def gen_rand_str(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def build_one_task():
    window_name = gen_rand_str(8)
    result_str = gen_rand_str(32)
    return ManualStoppingEnvironment(
        tools={},
        scoreboard_manager=DefaultScoreboardManager(),
        prompt={
            "prompt": HINT_TRAIN.replace("<window_name>", window_name),
            window_name: HINT_STEP1.replace("<result_str>", result_str),
        },
        max_steps=5,
        gt_label=result_str
    )