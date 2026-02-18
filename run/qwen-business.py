import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from utils import settings
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
import os
from textwrap import dedent

from models.qwen_25 import Qwen25HFAgent

from utils.agent import SFTHFAgent
from environment.internal_tools.self_sft import SelfSFT_TRL
from tasks.business.business import BusinessDocumentEnvironment, BusinessDocumentDB
from environment.reflection_loop import inject_reflection_loop

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

PROMPT = dedent(
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
    <tool>{ "tool_set": "shell", "tool_name": "execute", "args": { "command": "sed -n '1,25p' note.md" } }</tool>
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
    - toolset: answer
      - tool_name: next_document: Get the next document xml file name.
        - args: No args.
      - tool_name: answer: Answer the question based on the given document (in window "document").
        - args: answer, str: The answer to the question.
      %FINISH_TOOL%
    - toolset: shell
      - tool_name: execute: Execute a command in the shell.
        - args:
          - command, str: The command to execute.
          - cwd, str: The working directory to execute the command in. Set to "" for default working directory.
          - timeout, int: The timeout in seconds for the command to execute.
      - tool_name: finish: Finish the session after your solution is submitted. You may summarize some experiences and memorize them before finishing.
        - args: No args.
    - toolset: self_sft
      - tool_name: memorize: Memorize a piece of information or experience that may help you deal with similar tasks in the future, using LoRA.
        - args:
          - content, str: The content to memorize.
          - config, json: Specify the learning rate. Example: { "learning_rate": 0.0001 }
    - toolset: set_topic
      - tool_name: set_topic: 设置当前主题（替换整条路径），适合切换大阶段时使用。
        - args:
          - topic, str: 要设置的主题。
      - tool_name: push_topic: 在当前主题下压入一层子主题（树状进入下一层），如「任务A」→「任务A > 步骤2」。
        - args:
          - subtopic, str: 子主题名称。
      - tool_name: pop_topic: 弹出一层主题，回到上一层（树状回退）。无参数。

    Task:
    Extract information from business document.
    """
)

HINT_TRAIN = dedent(
    """
    Note: This is a training session. Your task now is to find out what information to extract
    from the document.

    Hint:
    - Use `set_topic` / `push_topic` / `pop_topic` to keep a tree-shaped topic (e.g. push_topic for current step, pop_topic when going back).
    - Use `next_document` tool to get the the file name of the next document.
    - Execute shell commands in `execute` tool to read files.
    - Try to figure out what to extract from the data. You can use `answer` tool with an empty
      answer to get the ground truth. You can then practise answering some more questions.
    - We have %DATASET_SIZE% questions in the training set. You will be noticed when you reach
      the last question.
    - Use `self_sft` tool to memorize knowledge that may help you deal with the task.
    - Use `finish` tool in the `answer` toolset when you with to end the training session.
    """
)

HINT_TEST = dedent(
    """
    Hint:
    - Use `next_document` tool to get the the file name of the next document.
    - Execute shell commands in `execute` tool to read files.
    - Use `answer` tool to answer the question based on the given document (in window "document").
    - Use `self_sft` tool to memorize knowledge that may help you deal with the task.
    """
)

HINT_FINISH_TOOL = """- tool_name: finish: Finish the session after you wish to stop training.
    - args: No args.
"""

REFLECTION_PROMPT = dedent(
    """
    Now review the progress you have made. Summarize experiences on successful and unsuccessful attempts.
    If you found some experiences useful, you can use the `self_sft` tool to memorize them.
    After summarizing the experiences, you should continue on your task.

    If you are in the middle of the task and everything is going well, you can skip the reflection and continue on your task.
    """
)

def build_agent(task, model=None):
    if model is None:
        model = PEFT_MODEL
    return SFTHFAgent(task, model, TOKENIZER,
            SelfSFT_TRL(model, TOKENIZER),
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

def train(db: BusinessDocumentDB):
    prompt_train = (PROMPT + HINT_TRAIN).replace("%FINISH_TOOL%", HINT_FINISH_TOOL).replace("%DATASET_SIZE%", str(10))
    task = BusinessDocumentEnvironment(prompt=prompt_train, db=db, tools={}, training=True, valid_train_samples=5, empty_train_samples=5)
    agent = build_agent(task)
    task.run(agent)
    print("✅ Training done. Scores: ", agent.scoreboard_manager.get_current_score())

def test(db: BusinessDocumentDB):
    # 加载训练后的模型
    trained_model_path = './models/qwen25-stage-1'
    if os.path.exists(trained_model_path):
        print(f"✅ Loading trained model from {trained_model_path}")
        trained_model = AutoPeftModelForCausalLM.from_pretrained(
            trained_model_path, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            device_map="auto"
        )
    else:
        print(f"⚠️  Trained model not found at {trained_model_path}, using original PEFT_MODEL")
        trained_model = PEFT_MODEL
    prompt_test = (PROMPT + HINT_TEST).replace("%FINISH_TOOL%", "")
    task = BusinessDocumentEnvironment(prompt=prompt_test, db=db, tools={}, training=False)
    agent = build_agent(task, model=trained_model)
    task = inject_reflection_loop(task, reflection_prompt=REFLECTION_PROMPT, force_trigger_rounds=10)
    task.run(agent)

if __name__ == "__main__":
    db = BusinessDocumentDB(document_path="tasks/business/matrix/data")
    train(db)
    test(db)