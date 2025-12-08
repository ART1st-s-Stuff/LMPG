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

from models.qwen_25 import Qwen25HFAgent

from utils.agent import SFTHFAgent
from environment.internal_tools.self_sft import SelfSFT_TRL
from tasks.retrieval.retrieval import build_tasks
from peft import AutoPeftModelForCausalLM

#os.environ["WKV_MODE"] = "chunk"

#MODEL = AutoModelForCausalLM.from_pretrained("./models/arwkv", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
PEFT_MODEL = AutoPeftModelForCausalLM.from_pretrained("./models/arwkv-stage-1", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
TOKENIZER = AutoTokenizer.from_pretrained("./models/arwkv", trust_remote_code=True)

#MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
#TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

def build_agent(task):
    return Qwen25HFAgent(task, PEFT_MODEL, TOKENIZER,
            SelfSFT_TRL(PEFT_MODEL, TOKENIZER, peft_config=PEFT_MODEL.config),
            Qwen25HFAgent.Config(
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
    tasks = build_tasks(100, TOKENIZER)
    scores = []
    for i, task in enumerate(tasks):
        agent = build_agent(task)
        task.run(agent)

        #agent.sft_trainer.train([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}], {"learning_rate": 3e-5})
        scores.append(agent.scoreboard_manager.get_current_score())
        suc = len([i for i in scores if i > 0])
        if scores[-1] > 0:
            # The agent successfully completed the task.
            # Save the history chat to a file.
            print(f"✅ Agent successfully completed the task. Successful runs: {suc}/{i+1}", )
        else:
            print(f"❌ Agent failed to complete the task. Skipping training. Successful runs: {suc}/{i+1}")
    print(scores)