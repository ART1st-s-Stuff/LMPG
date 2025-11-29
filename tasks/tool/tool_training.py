from textwrap import dedent
import os
import json
from typing import Dict

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
    I have finished step 1, but I have forgotten the next step to do. I need to open the hint
    again to decide my next action.
    <tool>{ "context": "text-default-hint", "tool": "read" }</tool>
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
    - text-default-hint: This hint.
    - text-default-step1: The window you should open in step 1. It tells you what to do next.

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
    - text-default-hint: This hint.
    - window_list: List of all the windows.
    
    You also have access to another tools:
    - end: End the task.

    Your task:
    Step 1. Open the "text-default-step1" window.
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
    Continue to step 2: Use the "stop" tool to end the task. It has context "stop" and tool "stop".
    """
)

TASK_PROMPT = dedent(
    """
    Answer the question based on the given document (in window "document").

    Question: {question}
    """
)


class Stop(Toolset):
    def __init__(self):
        super().__init__()
        self.stopped = False
    
    @Toolset.structurized_tool()
    def stop(self, _scoreboard_manager: ScoreboardManager):
        """Use this tool to end the task."""
        self.stopped = True
        _scoreboard_manager.get_scoreboard().reward(1000)

class ManualStoppingEnvironment(Environment):
    def __init__(self, tools: Dict[str, Toolset], scoreboard_manager: ScoreboardManager, prompt: Dict[str, str], max_steps: int):
        stop_tool = Stop()
        super().__init__({ **tools, "stop": stop_tool }, scoreboard_manager, prompt, lambda x: stop_tool.stopped, max_steps)

def build_one_task():
    return ManualStoppingEnvironment(
        tools={},
        scoreboard_manager=DefaultScoreboardManager(),
        prompt={
            "hint": HINT_TRAIN,
            "step1": HINT_STEP1,
        },
        max_steps=5
    )