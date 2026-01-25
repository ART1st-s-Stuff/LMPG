import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from utils import settings
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import LoraConfig, get_peft_model
import os
from textwrap import dedent

from models.qwen_25 import Qwen25HFAgent

from utils.agent import SFTHFAgent
from environment.internal_tools.self_sft import SelfSFT_TRL
from tasks.leetcode.leetcode import LeetcodeEnvironment

from dotenv import load_dotenv
load_dotenv()

MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, torch_dtype="auto", device_map="auto")
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, device_map="auto")

LORA_CONFIG = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
            )
PEFT_MODEL = get_peft_model(MODEL, LORA_CONFIG)

AUXILLIARY_CHAT_1 = [
    {"role" : "assistant", "content": 'If I forget the initial prompt, then I should alway take a look at window "prompt".'}
]
AUXILLIARY_CHAT_2 = [
    {"role" : "assistant", "content": 'If I want to see all opened windows, then I should alway take a look at window "opened_windows".'}
]

HINT_TRAIN = dedent(
    """
    You are an agent in the training process. Your are learning to use tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call. You must wrap the tool call within <tool></tool>
    tags, and follow the following specification:
    <tool>{ "tool_set": "<toolset name>", "tool_name": "<tool name>", "args": (in json format, can be empty) }</tool>
    Each tool has a "tool_set" field refering to the toolset of the tool, and a "tool_name" field refering to the tool name.
    The "tool_set" and "tool_name" fields are case sensitive. You need to follow the task-specific instructions 
    of the schema of the "args" field.

    Example output 1:
    ```
    <think>I have finished step 1, but I have forgotten the original task. I need to open the prompt
    again to decide my next action.</think>
    <tool>{ "tool_set": "window", "tool_name": "read", "args": { "window_name": "prompt" } }</tool>
    ```
    Example output 2:
    ```
    <think>First look at the `note.md` file to see the information.</think>
    <tool>{ "tool_set": "shell-sandbox", "tool_name": "execute", "args": { "command": "sed -n '1,25p' note.md" } }</tool>
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

    You have a view of multiple windows. You can read the prompt through the windows.
    To access the windows, you can use the following parameters in the tool call:
    - View a window: <tool>{ "tool_set": "window", "tool_name": "read", "args": { "window_name": "<window name>" } }</tool>
    - Go to a specific segment: <tool>{ "tool_set": "window", "tool_name": "goto", "args": { "window_name": "<window name>", "segment_number": int } }</tool>

    You have access to the following windows:
    - text-internal-prompt: This prompt window.
    - text-internal-tools: Tool usages.
    - text-internal-window_list: List of all opened windows.

    You have access to the following toolsets to help you complete the task:
    - toolset: leetcode
      - tool_name: submit: Submit the solution to the leetcode problem.
        - args: No args.
    - toolset: shell-sandbox
      - tool_name: execute: Execute a command in the shell.
        - args:
          - command, str: The command to execute.
          - cwd, str: The working directory to execute the command in. Set to "" for default working directory.
          - timeout, int: The timeout in seconds for the command to execute.
      - tool_name: finish: Finish the session after your solution is submitted. You may summarize some experiences and memorize them before finishing.
        - args: No args.
    - toolset: self_sft
      - tool_name: memorize: Memorize a piece of information or experience that may help you deal with similar tasks in the future.
        - args:
          - content, str: The content to memorize.
    - toolset: set_topic
      - tool_name: set_topic: Set the topic of the context. This will be presented in the beginning of all conversation history.
        - args:
          - topic, str: The topic to set.

    Task:
    Complete the leetcode problem %QID% using c++. The workspace is a generated directory under `cpp/`. The problem description is `cpp/<directory>/question.md`.
    Put your solution in `cpp/<directory>/solution.cpp`.
    
    Hint:
    - Execute shell commands in `execute` tool to read or write files.
    - Find the directory first.
    - Read the question, then set up a plan in `cpp/<directory>/plan.md`.
    - Follow the plan to implement the solution. You can modify the plan if needed.
    - You can summarize anything that you learned which may help you handle similar tasks in the future, including experience on the task and using tools, then memorize it using `memorize` tool.
    - You can compile and test the solution using `execute` tool.
    - When you have finished the solution, submit it to the leetcode problem using `submit` tool.
    - Finish the task using `finish` tool when your solution is accepted.
    """
)

def build_agent(task):
    return SFTHFAgent(task, MODEL, TOKENIZER,
            SelfSFT_TRL(PEFT_MODEL, TOKENIZER),
            SFTHFAgent.Config(
                TELL_REWARD_AFTER_EACH_ROUND=True,
                AUTO_SFT=False,
                AUTO_SFT_CONFIG={
                    "learning_rate": 1e-6
                },
                GENERATION_CONFIG=GenerationConfig(
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.6
                )
            )
        )

def get_env(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        input(f"Please enter the value for {key}: ")
    return value

if __name__ == "__main__":
    login_credentials = {
        "LEETCODE_SESSION": get_env("LEETCODE_SESSION"),
        "LEETCODE_CSRFTOKEN": get_env("LEETCODE_CSRFTOKEN"),
        "LEETCODE_CFCLEARANCE": get_env("LEETCODE_CFCLEARANCE")
    }
    qids = [ "44", "65", "126", "132" ]
    scores = []
    for qid in qids:
        prompt = HINT_TRAIN.replace("%QID%", qid)
        task = LeetcodeEnvironment(prompt=prompt, qid=qid, login_credentials=login_credentials, tools={})
        agent = build_agent(task)
        task.run(agent)

        #agent.sft_trainer.train([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], {"learning_rate": 3e-5})
        scores.append(agent.scoreboard_manager.get_current_score())
        if scores[-1] > 0:
            # The agent successfully completed the task.
            # Save the history chat to a file.
            print("✅ Agent successfully completed the task. Training the agent...")
            agent.sft_trainer.train(agent.history_chat, {"learning_rate": 3e-5})
            if len([ x for x in scores if x > 0 ]) > 14:
                print(scores)
                print("Stage 1 finished.")
                PEFT_MODEL.save_pretrained('./models/qwen25-stage-1')
                break
        else:
            print("❌ Agent failed to complete the task. Skipping training.")