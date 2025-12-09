from textwrap import dedent
import os
import json
from typing import Dict
import random
import string

from utils.tool import Toolset
from utils.environment import Environment
from utils.scoring import DefaultScoreboardManager, ScoreboardManager



# HINT_RWKV_OFFICIAL = dedent(
#     """
#     You are an agent in the training process. Your are learning to use tools efficiently.

#     You can interact with the environment in a turn-based manner. In each turn, your
#     output can contain at most 1 tool call.

#     You may also choose not to use any tools. All your output other than the tool call
#     will be coonsidered as your thinking process.

#     You have a view of multiple windows. You can read the document and prompts through
#     the windows. To access the windows, you can use the following parameters in the tool call:
#     - View a window: { "context": "window name", "tool": "read" }
#     - Go to a specific segment: { "context": "window name", "tool": "goto", "args": { "segment_number": int } }

#     In this stage, you have access to the following windows:
#     - text-default-prompt: This prompt.
#     - window_list: List of all the windows.
    
#     You also have access to another tools:
#     - end: End the task.

#     Your task:
#     Step 1. Open the "<window_name>" window.
#     Step 2. Use the end tool to end the task.

#     Tool call specification:
#     {
#         "tools": [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "tool",
#                     "description": "Execute a tool call",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {
#                             "context": {
#                                 "type": "string",
#                                 "description": "The context of the tool call"
#                             },
#                             "tool": {
#                                 "type": "string",
#                                 "description": "The name of the tool to execute"
#                             },
#                             "args": {
#                                 "type": "object",
#                                 "description": "The arguments of the tool call"
#                             },
#                         },
#                         "required": ["context", "tool"]
#                     }
#                 }
#             }
#         ]
#     }
#     """
# )

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

def build_one_task(hint: str, window_name=None, result_str=None):
    window_name = window_name or gen_rand_str(4)
    result_str = result_str or gen_rand_str(8)
    return ManualStoppingEnvironment(
        tools={},
        scoreboard_manager=DefaultScoreboardManager(),
        prompt={
            "prompt": hint.replace("<window_name>", window_name),
            window_name: HINT_STEP1.replace("<result_str>", str(result_str)),
        },
        max_steps=5,
        gt_label=result_str
    )