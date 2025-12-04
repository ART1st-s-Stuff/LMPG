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

from utils.agent import SFTHFAgent
from environment.internal_tools.self_sft import SelfSFT_TRL
from tasks.tool.tool_training import build_one_task

os.environ["WKV_MODE"] = "chunk"

MODEL = AutoModelForCausalLM.from_pretrained("./models/arwkv", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
TOKENIZER = AutoTokenizer.from_pretrained("./models/arwkv", trust_remote_code=True)

LORA_CONFIG = LoraConfig(
                r=8,
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

#MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
#TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

if __name__ == "__main__":
    tasks = [build_one_task() for _ in range(100)]
    scores = []
    for task in tasks:
        agent = SFTHFAgent(task, MODEL, TOKENIZER,
            SelfSFT_TRL(PEFT_MODEL, TOKENIZER, peft_config=LORA_CONFIG),
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
        task.run(agent)

        #agent.sft_trainer.train([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], {"learning_rate": 3e-5})
        scores.append(agent.scoreboard_manager.get_current_score())
        if scores[-1] > 0:
            # The agent successfully completed the task.
            # Save the history chat to a file.
            print("✅ Agent successfully completed the task. Training the agent...")
            agent.sft_trainer.train(agent.history_chat, {"learning_rate": 3e-5})
            if len(scores) > 20:
                print(scores)
                print("Training on aux dataset...")
                for _ in range(20):
                    agent.sft_trainer.train(AUXILLIARY_CHAT_1, {"learning_rate": 3e-5})
                    agent.sft_trainer.train(AUXILLIARY_CHAT_2, {"learning_rate": 3e-5})
                PEFT_MODEL.save_pretrained('./models/arwkv-stage-1')
                break
        else:
            print("❌ Agent failed to complete the task. Skipping training.")