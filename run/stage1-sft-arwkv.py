import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import json
import os
from textwrap import dedent
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import LoraConfig, get_peft_model

from models.qwen_25 import Qwen25HFAgent

from utils import settings
from utils.agent import SFTHFAgent
from environment.internal_tools.self_sft import SelfSFT_TRL
from tasks.tool.tool_training import build_one_task

os.environ["WKV_MODE"] = "chunk"

MODEL = AutoModelForCausalLM.from_pretrained("./models/arwkv", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
TOKENIZER = AutoTokenizer.from_pretrained("./models/arwkv", trust_remote_code=True)
# MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
# TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

LORA_CONFIG = LoraConfig(
                r=32,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["receptance", "value"]
            )
PEFT_MODEL = get_peft_model(MODEL, LORA_CONFIG)

AUXILLIARY_CHAT_1 = [
    {"role" : "assistant", "content": 'If I forget the initial prompt, then I should alway take a look at window "text-default-prompt".'}
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
    <tool>{ "context": "context name", "tool": "tool name", "args": (in json format, can be empty) }</tool>
    Each tool has a "context" field refering to the context of the tool, and a "tool" field refering to the tool name.
    The "context" and "tool" fields are case sensitive. You need to follow the task-specific instructions 
    of the schema of the "args" field.

    Example output 1:
    ```
    <think>I have finished step 1, but I have forgotten the original task. I need to open the prompt
    again to decide my next action.</think>
    <tool>{ "context": "text-default-prompt", "tool": "read", "args": {} }</tool>
    ```
    Example output 2:
    ```
    <think>I have found the answer 29380. Now I output the answer.</think>
    <tool>{ "context": "interact", "tool": "output", "args": { "content" : 29380 } }</tool>
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

    You have a view of multiple windows. You can read the document and prompts through
    the windows. To access the windows, you can use the following parameters in the tool call:
    - View a window: <tool>{ "context": "window name", "tool": "read", "args": {} }</tool>
    - Go to a specific segment: <tool>{ "context": "window name", "tool": "goto", "args": { "segment_number": int } }</tool>

    In this stage, you have access to the following windows:
    - <window_name>: The window you should open for your task. Contains prompt for what to do next.

    Task:
    Step 1. Open the "<window_name>" window, and follow the instruction there.
    """
)

def build_agent(task):
    return SFTHFAgent(task, MODEL, TOKENIZER,
            SelfSFT_TRL(PEFT_MODEL, TOKENIZER, {"label_names":["labels"]}),
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

if __name__ == "__main__":
    tasks2 = [build_one_task(HINT_TRAIN) for x in range(200)]
    scores = []
    for task in tasks2:
        agent = build_agent(task)
        task.run(agent)

        #agent.sft_trainer.train([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], {"learning_rate": 3e-5})
        scores.append(agent.scoreboard_manager.get_current_score())
        if scores[-1] > 0:
            # The agent successfully completed the task.
            # Save the history chat to a file.
            print("✅ Agent successfully completed the task. Training the agent...")
            agent.sft_trainer.train(agent.history_chat, {"learning_rate": 3e-5})
            if len([ x for x in scores if x > 0 ]) > 10:
                print(scores)
                print("Stage 1.1 finished.")
                # for _ in range(10):
                #     agent.sft_trainer.train(AUXILLIARY_CHAT_1, {"learning_rate": 3e-5})
                #     agent.sft_trainer.train(AUXILLIARY_CHAT_2, {"learning_rate": 3e-5})
                PEFT_MODEL.save_pretrained('./models/arwkv-stage-1')
                break
        else:
            print("❌ Agent failed to complete the task. Skipping training.")