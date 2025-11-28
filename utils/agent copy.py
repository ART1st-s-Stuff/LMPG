from typing import Dict, Optional, Tuple, List, Callable
from dataclasses import dataclass
import logging
import torch
from transformers.generation import GenerationMixin, GenerationConfig
from transformers import AutoTokenizer
from trl import SFTConfig
from abc import ABC, abstractmethod

from environment.internal_tools.self_sft import SelfSFT, SelectedSFTConfig
from .environment import Environment
from .scoring import ScoreboardManager
from .tool import Toolset, parse_llm_output
from .exceptions import ToolCallException, ContextNotExistException
from .text import TextWindow, text_window, SegmentTextWindow, FileTextWindow
from . import settings

logging.basicConfig(level=logging.DEBUG)

class Agent(ABC):
    environment: Environment
    scoreboard_manager: ScoreboardManager
    toolsets: Dict[str, Toolset]
    default_windows: Dict[str, TextWindow]
    open_windows: Dict[str, TextWindow]
    history_state: Optional[torch.Tensor]

    @dataclass
    class Config:
        ENABLE_THINKING: bool = False
        TELL_REWARD_AFTER_EACH_ROUND: bool = False
        GENERATION_CONFIG: GenerationConfig = GenerationConfig()

    def __init__(self, environment: Environment, model: GenerationMixin, config: Config):
        self.config = config
        self.model = model

        self.pre_tool_call_hooks : List[Callable[['Agent', str, Optional[str], Optional[str], Optional[str]], Optional[str]]] = []
        self.post_tool_call_hooks : List[Callable[['Agent', str, str, str, str], Optional[str]]] = []
        self.set_environment(environment)

    def set_environment(self, environment: Environment):
        self.history_state = None
        self.open_windows = {}
        self.default_windows = {}
        self.environment = environment
        self.scoreboard_manager = environment.scoreboard_manager
        self.toolsets = environment.tools
        
        # Set default windows: Prompt, tools, scoreboard, opened windows list
        for key, value in environment.prompt.items():
            self.default_windows[f"prompt-{key}"] = text_window(
                value,
                window_id=key,
                interface_prefix="default",
                window_type="segment"
            )
        self.default_windows["tools"] = text_window(
            self.environment.tools_hint,
            window_id="tools",
            interface_prefix="default",
            window_type="segment"
        )
        self.default_windows["scoreboard"] = self.scoreboard_manager.get_output_window()
        self._update_window_list()

    # @abstractmethod
    # def pause(self):
    #     raise NotImplementedError()
    def _update_window_list(self):
        _default_windows = '\n'.join(self.default_windows.keys())
        _opened_windows = '\n'.join(self.open_windows.keys())
        self.default_windows["window_list"] = text_window(
            [
                f"Window operation hints: {SegmentTextWindow.hint}\n{FileTextWindow.hint}",
                f"Default windows:\n{_default_windows}",
                f"Opened windows:\n{_opened_windows}"
            ],
            window_id="window_list",
            interface_prefix="default",
            window_type="segment"
        )
    
    def forward(self, input: torch.Tensor, config) -> Tuple[torch.Tensor, torch.Tensor]:
        """Low-level forward pass."""
        if self.history_state is not None:
            full_input = torch.cat([self.history_state, input], dim=0).to(self.model.device)
        else:
            full_input = input.to(self.model.device)
        generated = self.model.generate(
            full_input,
            attention_mask=torch.ones_like(full_input).to(self.model.device),
            generation_config=config,
            max_new_tokens=4096
        )
        self.history_state = generated[0]
        output = generated[0][full_input.shape[1]:]
        return output

    def step(self, input: Optional[str | Dict[str, str]] = None) -> Dict[str, str]:
        tokenized_input = self.tokenize(input)
        output_tokens = self.forward(tokenized_input, self.config.GENERATION_CONFIG)
        output = self.detokenize(output_tokens)
        tool_call_result = self._post_step(output)
        return tool_call_result

    def add_pre_tool_call_hook(self, hook: Callable[['Agent', str, Optional[str], Optional[str], Optional[str]], Optional[str]]):
        self.pre_tool_call_hooks.append(hook)

    def add_post_tool_call_hook(self, hook: Callable[['Agent', str, str, str, str], Optional[str]]):
        self.post_tool_call_hooks.append(hook)

    def _pre_tool_call_hook(self, model_output: str, context: Optional[str], tool: Optional[str], tool_input: Optional[str]) -> Optional[str]:
        """Hook for tool calling. If a string is returned, it will be used as the
        tool call output and skips the default tool calling process."""
        for hook in self.pre_tool_call_hooks:
            result = hook(self, model_output=model_output, context=context, tool=tool, tool_input=tool_input)
            if result is not None:
                return result
        return None

    def _post_tool_call_hook(self, context: str, tool: str, tool_input: str, tool_output: str) -> Optional[str]:
        """Hook for post-tool calling. If a string is returned, it will be used
        directly as the tool call output."""
        for hook in self.post_tool_call_hooks:
            result = hook(self, context=context, tool=tool, tool_input=tool_input, tool_output=tool_output)
            if result is not None:
                return result
        return tool_output

    @abstractmethod
    def detokenize(self, output: torch.Tensor) -> str:
        """Parse the generic model output into a string."""
        raise NotImplementedError()

    @abstractmethod
    def tokenize(self, input: str | Dict[str, str]) -> torch.Tensor:
        """Parse the tool output into a generic model input."""
        raise NotImplementedError()

    def _post_step(self, model_output: str) -> Dict[str, str]:
        logging.debug(f"Model output:\n{model_output}")
        tool_output = self._call_tool(model_output)
        if self.config.TELL_REWARD_AFTER_EACH_ROUND:
            current_round_reward = self.scoreboard_manager.get_current_step_output()
            output = {
                "ai" : model_output,
                "environment" : tool_output,
                "reward" : current_round_reward,
            }
        else:
            output = {
                "ai" : model_output,
                "environment" : tool_output,
            }
        output = { key: value for key, value in output.items() if value is not None }
        return output

    def _call_tool(self, output: str) -> Optional[str]:
        # Parse output
        try:
            context, tool, tool_input = parse_llm_output(output)
            # First execute hook.
            tool_output = self._pre_tool_call_hook(output, context, tool, tool_input)
            if tool_output is not None:
                return tool_output
            if context is None:
                # No tool call in this round. Skip.
                return None
            # Handle window operations
            elif context in self.open_windows:
                ctx = self.open_windows[context]
                # Only open_windows can be closed
                if tool == "close":
                    del self.open_windows[context]
                    tool_output = f"Closed window {context}."
                else:
                    tool_output = ctx.invoke(tool, tool_input)
            elif context in self.default_windows:
                ctx = self.default_windows[context]
                tool_output = ctx.invoke(tool, tool_input)
            # Not internal tools. Try context tools.
            elif context in self.toolsets:
                ctx = self.toolsets[context]
                tool_output = ctx.invoke(context, tool, tool_input)
                # If the output is a text window
                if isinstance(tool_output, TextWindow):
                    if not tool_output.volatile:
                        self.open_windows[tool_output.window_name] = tool_output
                    else:
                        # TODO: Handle volatile windows.
                        ...
                    tool_output = tool_output.read()
            else:
                raise ContextNotExistException(context)
            self._update_window_list()
            tool_output = self._post_tool_call_hook(context, tool, tool_input, tool_output)
            return tool_output
        except ToolCallException as e:
            self.scoreboard_manager.get_scoreboard(Agent).reward(e.penalty, str(e))
            return f"{str(e)}"

    def clear_state(self):
        self.history_state = None

    # def interrupt(self, input: str):
    #     self.pause()
    #     self.forward(input)

class DefaultTokenizerAgent(Agent):
    def __init__(self, environment: Environment, model: GenerationMixin, tokenizer: AutoTokenizer, config: Agent.Config):
        super().__init__(environment, model, config)
        self.tokenizer = tokenizer

    def to_chat_format(self, input: str | Dict[str, str]) -> str:
        if isinstance(input, dict):
            input = [{"role": "assistant", "content": input["ai"]}]
            user = ""
            if "environment" in input:
                user += input["environment"] + "\n\n"
            if "reward" in input:
                user += input["reward"]
            if user != "":
                input.append({"role": "user", "content": user})
        else:
            input = [{"role": "user", "content": input}]
        return input

    def tokenize(self, input: str | Dict[str, str]) -> torch.Tensor:
        input = self.to_chat_format(input)
        text = self.tokenizer.apply_chat_template(
            input,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config.ENABLE_THINKING  # Default is True, set to False to disable thinking
        )
        logging.debug(f"Stepping with input:\n{text}")
        return self.tokenizer([text], return_tensors="pt").to(self.model.device).input_ids

    def detokenize(self, output: torch.Tensor) -> str:
        return self.tokenizer.batch_decode([output], skip_special_tokens=True)[0]

class SFTAgent(DefaultTokenizerAgent):
    @dataclass
    class Config(Agent.Config):
        SFT_TRAINER: SelfSFT = None
        AUTO_SFT: bool = False
        AUTO_SFT_CONFIG: SelectedSFTConfig = None

    def __init__(self, environment: Environment, model: GenerationMixin, tokenizer, config: Config):
        super().__init__(environment, model, tokenizer, config)
        self.sft_trainer = config.SFT_TRAINER
        self.toolsets["self_sft"] = self.sft_trainer
        self.add_post_tool_call_hook(self._self_sft)

    @staticmethod
    def _self_sft(instance: "SFTAgent", context: str, tool: str, tool_input: str, tool_output: str):
        if instance.sft_trainer.changed:
            instance.update_model(instance.sft_trainer.model)
            instance.sft_trainer.reset_changed()

    def _post_step(self, model_output: str) -> Dict[str, str]:
        result = super()._post_step(model_output)
        if self.config.AUTO_SFT:
            self.sft_trainer.train(self.to_chat_format(result), self.config.AUTO_SFT_CONFIG)
            self.model = self.sft_trainer.model
            self.sft_trainer.reset_changed()
        return result

    def update_model(self, model):
        self.model = model