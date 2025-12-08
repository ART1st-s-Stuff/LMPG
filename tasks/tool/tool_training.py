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
    output can contain at most 1 tool call.

    Example output 1:
    ```
    <think>I have finished step 1, but I have forgotten the original task. I need to open the prompt
    again to decide my next action.</think>
    <tool_call>...</tool_call>
    ```
    Example output 2:
    ```
    <think>I have found the answer 29380. Now I output the answer.</think>
    <tool_call>...</tool_call>
    ```
    Example output 3:
    ```
    <think>We should run the program again and expect a successful output.
    But the log suggests we cannot connect to the API. However, we have previously set the
    proxy and curl test to the API is fine. Perhaps it is the due to library requires
    explicit proxy settings.
    I should try to figure out if the proxy settings is correct in the next step.</think>
    ```

    You may also choose not to use any tools. All your output other than the tool call
    will be coonsidered as your thinking process.

    In this stage, you have access to the following windows:
    - <window_name>: The window you should open for your task. Contains prompt for what to do next.
    - prompt: This window.

    Task:
    Step 1. Open the "<window_name>" window, and follow the instruction there.
    """
)

# HINT_TRAIN = dedent(
#     """
#     You are an agent in the training process. Your are learning to use tools efficiently.

#     You can interact with the environment in a turn-based manner. In each turn, your
#     output can contain at most 1 tool call. You must wrap the tool call within <tool></tool>
#     tags, and follow the following specification:
#     <tool>{ "context": "context name", "tool": "tool name", "args": (in json format, can be empty) }</tool>
#     The "context" and "tool" fields are case sensitive. You need to follow the task-specific instructions 
#     of tool args schema.

#     Example output 1:
#     ```
#     <think>I have finished step 1, but I have forgotten the original task. I need to open the prompt
#     again to decide my next action.</think>
#     <tool>{ "context": "text-default-prompt", "tool": "read", "args": {} }</tool>
#     ```
#     Example output 2:
#     ```
#     <think>I have found the answer 29380. Now I output the answer.</think>
#     <tool>{ "context": "interact", "tool": "output", "args": { "content" : 29380 } }</tool>
#     ```
#     Example output 3:
#     ```
#     <think>We should run the program again and expect a successful output.
#     But the log suggests we cannot connect to the API. However, we have previously set the
#     proxy and curl test to the API is fine. Perhaps it is the due to library requires
#     explicit proxy settings.
#     I should try to figure out if the proxy settings is correct in the next step.</think>
#     ```

#     You may also choose not to use any tools. All your output other than the tool call
#     will be coonsidered as your thinking process.

#     You have a view of multiple windows. You can read the document and prompts through
#     the windows. To access the windows, you can use the following parameters in the tool call:
#     - View a window: <tool>{ "context": "window name", "tool": "read", "args": {} }</tool>
#     - Go to a specific segment: <tool>{ "context": "window name", "tool": "goto", "args": { "segment_number": int } }</tool>

#     In this stage, you have access to the following windows:
#     - text-default-<window_name>: The window you should open for your task. Contains prompt for what to do next.

#     Task:
#     Step 1. Open the "text-default-<window_name>" window, and follow the instruction there.
#     """
# )

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
    Usage: <tool>{ "context": "output", "tool": "output", "args": { "content" : "..." } }</tool>
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
        if content == str(self.gt):
            _scoreboard_manager.get_scoreboard().reward(1000)

class ManualStoppingEnvironment(Environment):
    def __init__(self, tools: Dict[str, Toolset], scoreboard_manager: ScoreboardManager, prompt: Dict[str, str], max_steps: int, gt_label: str):
        output_tool = Output(gt_label)
        super().__init__({ **tools, "output": output_tool }, scoreboard_manager, prompt, lambda x: output_tool.stopped, max_steps)

def gen_rand_str(length):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def build_one_task(window_name=None, result_str=None):
    window_name = window_name or gen_rand_str(4)
    result_str = result_str or gen_rand_str(8)
    return ManualStoppingEnvironment(
        tools={},
        scoreboard_manager=DefaultScoreboardManager(),
        prompt={
            "prompt": HINT_TRAIN.replace("<window_name>", window_name),
            window_name: HINT_STEP1.replace("<result_str>", str(result_str)),
        },
        max_steps=5,
        gt_label=result_str
    )