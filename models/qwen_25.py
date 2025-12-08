from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import json
import re

import torch
from transformers.generation import GenerationMixin, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.agent import HFMixin, SFTAgent, SFTHFAgent
from utils.exceptions import ToolCallException, ContextNotExistException
from utils.environment import Environment
from environment.internal_tools.self_sft import SelfSFT, SelectedSFTConfig

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "window",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The window name.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["read", "goto"],
                        "description": "The action to the window. `read` returns the current segment of the window; `goto` allows to jump to a segment.",
                    },
                    "args": {
                        "type": "object",
                        "description": 'Optional additional argument. Required only by action `goto`, should be { "segment_number": int }.'
                    }
                },
                "required": ["name", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool",
            "description": "Use external tools provided by the environment. Read the prompt for usage instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The context name of the tool.",
                    },
                    "tool": {
                        "type": "string",
                        "description": "The name of the tool.",
                    },
                    "args": {
                        "type": "object",
                        "description": "Tool argument. Refer to tool usage instruction for details."
                    }
                },
                "required": ["context", "tool"],
            },
        },
    }
]

class Qwen25Mixin(HFMixin):
    def tokenize(self, input: str | Dict[str, str]) -> torch.Tensor:
        input = self._to_chat_format(input)
        text = self.tokenizer.apply_chat_template(
            input,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            **self.hf_config.CHAT_TEMPLATE_ARGS
        )
        return input, self.tokenizer([text], return_tensors="pt").input_ids[0]

class Qwen25HFAgent(Qwen25Mixin, SFTAgent[AutoModelForCausalLM]):
    @dataclass
    class Config(SFTAgent.Config, HFMixin.Config):
        pass

    def __init__(self, environment: Environment, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sft_trainer: SelfSFT, config: Config):
        super().__init__(model=model, tokenizer=tokenizer, hf_config=config, environment=environment, sft_trainer=sft_trainer, config=config)

    def _forward(self, input: str | Dict[str, str]) -> str:
        return Qwen25Mixin._forward(self, input)

def parse_llm_output(output: str) -> Tuple[Optional[str], Optional[str], Optional[str | Dict[str, Any]]]:
    regex = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    match : List[str] = regex.findall(output)
    if len(match) > 1:
        raise MultipleToolCallException()
    if len(match) == 0:
        return None, None, None
    tool_call = match[0]
    try:
        tool_call_json = json.loads(tool_call)
        if tool_call_json["name"] == "window":
            return "text-default-" + tool_call_json["arguments"]["name"], tool_call_json["arguments"]["action"], tool_call_json["arguments"].get("args", {})
        else:
            assert tool_call_json["name"] == "tool"
            assert isinstance(tool_call_json["arguments"]["context"], str)
            assert isinstance(tool_call_json["arguments"]["tool"], str)
            return tool_call_json["arguments"]["context"], tool_call_json["arguments"]["tool"], tool_call_json["arguments"].get("args", {})
    except Exception as e:
        print(e)
        raise InvalidToolCallJSONException()
    