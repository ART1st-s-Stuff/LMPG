from textwrap import dedent
import os
import json

from utils.tool import Toolset
from utils.environment import ManualStoppingEnvironment
from utils.scoring import DefaultScoreboardManager, Scoreboard


HINT_TRAIN = dedent(
    """
    You are an agent in the training process. Your are learning to use tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call.

    Tool call specification:
    <tool_call>{ "context": "context name", "tool": "tool name", "args": (optional, in json format) }</tool_call>

    You may also choose not to use any tools. All your output other than the tool call
    will be coonsidered as your thinking process.

    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following parameters in the tool call:
    - View a window: { "context": "window name", "tool": "read" }
    - Go to a specific segment: { "context": "window name", "tool": "goto", "args": { "segment_number": int } }

    In this stage, you have access to the following windows:
    - TEXT-default-hint: This hint.
    - window_list: List of all the windows.
    
    You also have access to another tools:
    - end: End the task.

    Your task:
    Step 1. Open the "TEXT-default-step1" window.
    Step 2. Use the end tool to end the task.
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
    - TEXT-default-hint: This hint.
    - window_list: List of all the windows.
    
    You also have access to another tools:
    - end: End the task.

    Your task:
    Step 1. Open the "TEXT-default-step1" window.
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
    Continue to step 2: Use the end tool to end the task.
    """
)

TASK_PROMPT = dedent(
    """
    Answer the question based on the given document (in window "document").

    Question: {question}
    """
)

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