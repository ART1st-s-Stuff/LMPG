import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from utils import settings
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
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
# TODO
GRPO_CONFIG = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    remove_unused_columns=False,
    logging_steps=1,
)
PEFT_MODEL = get_peft_model(MODEL, LORA_CONFIG)

#MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
#TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

if __name__ == "__main__":
    tasks = [build_one_task() for _ in range(100)]
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
        if agent.scoreboard_manager.get_current_score() > 0:
            # The agent successfully completed the task.
            # Save the history chat to a file.
            print("✅ Agent successfully completed the task. Training the agent...")
            agent.sft_trainer.train(agent.history_chat, {"learning_rate": 3e-5})
            PEFT_MODEL.save_pretrained('./models/arwkv-stage-1')
        else:
            print("❌ Agent failed to complete the task. Skipping training.")