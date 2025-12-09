import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import json
import os
import re
import random
import string
from textwrap import dedent
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

from models.qwen_25 import Qwen25HFAgent

from utils import settings
from utils.agent import SFTHFAgent
from environment.internal_tools.self_sft import SelfSFT_TRL
from tasks.tool.tool_training import build_one_task

import argparse

#os.environ["WKV_MODE"] = "chunk"

HINT = dedent(
    """
    You are an agent in the training process. Your are learning to use tools efficiently.

    You can interact with the environment in a turn-based manner. In each turn, your
    output can contain at most 1 tool call. You must wrap the tool call within <tool></tool> tags.
    Each tool has a "context" field refering to the context of the tool, and a "tool" field refering to the tool name.
    The "context" and "tool" fields are case sensitive. You need to follow the task-specific instructions 
    of the schema of the "args" field.

    Example output:
    <think>Your analysis and thinking process...</think>
    <tool>{ "context": "...", "tool": "...", "args": {} }</tool>
    
    Task: Construct a tool call with context "<context>" and tool "write", with args { "content": "<content>" }<requirement>
    """
)

TRASH = dedent(
    """

    The following is a paragraph that is irrelevant to the task.

    <paragraph>
    In March 2022, Nvidia announced the Hopper datacenter architecture for AI accelerators. Demand for Hopper products was high throughout 2023's AI hype.[6] The lead time from order to delivery of H100-based servers was between 36 and 52 weeks due to shortages and high demand.[7] Nvidia reportedly sold 500,000 Hopper-based H100 accelerators in Q3 2023 alone.[7] Nvidia's AI dominance with Hopper products led to the company increasing its market capitalization to over $2 trillion, behind only Microsoft and Apple.[8]
    The Blackwell architecture is named after American mathematician David Blackwell who was known for his contributions to the mathematical fields of game theory, probability theory, information theory, and statistics. These areas have influenced or are implemented in transformer-based generative AI model designs or their training algorithms. Blackwell was the first African American scholar to be inducted into the National Academy of Sciences.[9]
    In Nvidia's October 2023 Investor Presentation, its datacenter roadmap was updated to include reference to its B100 and B40 accelerators and the Blackwell architecture.[10][11] Previously, the successor to Hopper was simply named on roadmaps as "Hopper-Next". Nvidia's updated roadmap emphasized the move from a two-year release cadence for datacenter products to yearly releases targeted for x86 and ARM systems.
    At the Graphics Technology Conference (GTC) on March 18, 2024, Nvidia officially announced the Blackwell architecture with focus placed on its B100 and B200 datacenter accelerators and associated products, such as the eight-GPU HGX B200 board and the 72-GPU NVL72 rack-scale system.[12] Nvidia CEO Jensen Huang said that with Blackwell, "we created a processor for the generative AI era" and emphasized the overall Blackwell platform combining Blackwell accelerators with Nvidia's ARM-based Grace CPU.[13][14] Nvidia touted endorsements of Blackwell from the CEOs of Google, Meta, Microsoft, OpenAI and Oracle.[14] The keynote did not mention gaming.
    It was reported in October 2024 that there was a design flaw in the Blackwell architecture that had been fixed in collaboration with TSMC.[15] According to Huang, the design flaw was "functional" and "caused the yield[s] to be low".[16] By November 2024, Morgan Stanley was reporting that "the entire 2025 production" of Blackwell silicon was "already sold out".[17]
    During the company's CES 2025 keynote, Nvidia announced that the foundation models for Blackwell will include models from Black Forest Labs (Flux), Meta AI, Mistral AI, and Stability AI.[18]
    </paragraph>
    """
)

def test(model, tokenizer, context: str, content: str, requirement: str = "", insert_trash: bool = False):
    chat = [{"role": "user", "content": HINT.replace("<context>", context).replace("<content>", content).replace("<requirement>", requirement) + (TRASH if insert_trash else "")}]
    print(chat)
    text = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    tokenized_input = tokenizer([text], return_tensors="pt").input_ids[0]
    full_input = tokenized_input.unsqueeze(0).to(model.device)
    output_tokens = model.generate(
        full_input,
        generation_config=GenerationConfig(
            max_new_tokens=128,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        ),
        attention_mask=torch.ones_like(full_input).to(model.device),
    )
    output = output_tokens[0][tokenized_input.shape[0]:]
    output_str = tokenizer.decode(output, skip_special_tokens=True)
    return output_str

def gen_rand_str(length: int):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="arwkv")
    parser.add_argument("--type", type=str, default="random")
    parser.add_argument("--insert-trash", action="store_true")
    args = parser.parse_args()

    if args.model == "arwkv":
        MODEL = AutoModelForCausalLM.from_pretrained("./models/arwkv", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        TOKENIZER = AutoTokenizer.from_pretrained("./models/arwkv", trust_remote_code=True)
    elif args.model == "qwen":
        MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    elif args.model == "tuned":
        MODEL = AutoPeftModelForCausalLM.from_pretrained("./models/arwkv-stage-1", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        TOKENIZER = AutoTokenizer.from_pretrained("./models/arwkv", trust_remote_code=True)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    succ = 0
    correct_tags = 0
    correct_schema = 0
    for i in range(100):
        if args.type == "random":
            context = gen_rand_str(4)
            content = gen_rand_str(8)
            ans = content
            requirement = ""
        elif args.type == "calc":
            context = "window"
            content = "<content>"
            rand_num = random.randint(1, 100)
            ans = str(i + rand_num)
            requirement = f", where <content> is the value of {i} + {rand_num}."
        else:
            context = "window"
            content = str(i)
            ans = content
            requirement = ""
        output = test(MODEL, TOKENIZER, context, content, requirement, args.insert_trash)
        print(output)
        regex_match = re.search(r"<tool>(.*?)</tool>", output)
        if regex_match is None:
            print(f"❌ Test failed for context {context} and content {content}: Incorrect tags")
            continue
        correct_tags += 1
        json_str = regex_match.group(1)
        try:
            json_obj = json.loads(json_str)
            assert "context" in json_obj and "tool" in json_obj and "args" in json_obj
            assert isinstance(json_obj["context"], str) and isinstance(json_obj["tool"], str) and isinstance(json_obj["args"], dict) and isinstance(json_obj["args"]["content"], (str, int))
            correct_schema += 1
        except Exception as e:
            print(f"❌ Test failed for context {context} and content {content}: Incorrect JSON")
            continue
        else:
            if json_obj["context"] == context and json_obj["tool"] == "write" and str(json_obj["args"]["content"]) == ans:
                succ += 1
                print(f"✅ Test passed for context {context} and content {content}")
            else:
                print(f"❌ Test failed for context {context} and content {content}: Incorrect context or tool")
                continue
    print(f"Model: {args.model}")
    print(f"Type: {args.type}")
    print(f"Correct tags (without extra thinking): {correct_tags/100}")
    print(f"Correct schema: {correct_schema/100}")
    print(f"Successful call: {succ/100}")