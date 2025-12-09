from typing import Dict, List, Tuple, Callable, Optional, Any, TypeVar, Generic, TypedDict
from dataclasses import dataclass, field
import logging
import torch
from transformers.generation import GenerationMixin, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod

from environment.internal_tools.self_sft import SelfSFT, SelectedSFTConfig
from .environment import Environment
from .scoring import ScoreboardManager
from .tool import Toolset, parse_llm_output
from .exceptions import ToolCallException, ContextNotExistException
from .text import TextWindow, text_window, SegmentTextWindow, FileTextWindow
from . import settings

logging.basicConfig(level=logging.DEBUG)

class AgentForwardContent(TypedDict):
    """Content of the agent forward pass.

    Args:
        ai: The AI generated output. If the state is well managed, this field can be ignored.
        environment: The output of the tool calls.
        reward: Reward of the current step.
    """
    ai: str
    environment: Optional[str]
    reward: Optional[str]

class StateManagerMixin(ABC):
    @abstractmethod
    def _forward(self, input: str | AgentForwardContent) -> str:
        """Low-level forward pass."""
        raise NotImplementedError()
        
    @abstractmethod
    def clear_state(self):
        """Clear the history state of the agent."""
        raise NotImplementedError()

class OutputLengthPenaltyMixin(ABC):
    def __init__(self, output_length_penalty: float, max_output_length: int, **kwargs):
        super().__init__(**kwargs)
        self.output_length_penalty = output_length_penalty
        self.max_output_length = max_output_length
        self.add_pre_tool_call_hook(self._output_length_penalty_hook)

    @staticmethod
    def _output_length_penalty_hook(instance: 'OutputLengthPenaltyMixin', model_output: str, context: Optional[str], tool: Optional[str], tool_input: Optional[str]) -> int:
        context_length = instance._calculate_output_length(model_output)
        if instance.max_output_length > 0 and context_length > instance.max_output_length:
            instance.scoreboard_manager.get_scoreboard().reward(instance.output_length_penalty, f"Output length exceeds the maximum length of {instance.max_output_length}.")

    @abstractmethod
    def _calculate_output_length(self, model_output: str) -> int:
        """Calculate the context length of the model_output."""
        raise NotImplementedError()

class Agent(StateManagerMixin):
    environment: Environment
    scoreboard_manager: ScoreboardManager
    toolsets: Dict[str, Toolset]
    default_windows: Dict[str, TextWindow]
    open_windows: Dict[str, TextWindow]
    config: 'Config'
    pre_tool_call_hooks : List[Callable[['Agent', str, Optional[str], Optional[str], Optional[str]], Optional[str]]]
    post_tool_call_hooks : List[Callable[['Agent', str, str, str, str], Optional[str]]]

    @dataclass
    class Config:
        TELL_REWARD_AFTER_EACH_ROUND: bool = field(default=False)

    def __init__(self, environment: Environment, config: Config):
        self.config = config
        self.pre_tool_call_hooks = []
        self.post_tool_call_hooks = []
        self.set_environment(environment)

    def set_environment(self, environment: Environment):
        self.open_windows = {}
        self.default_windows = {}
        self.environment = environment
        self.scoreboard_manager = environment.scoreboard_manager
        self.toolsets = environment.tools
        
        # Set default windows: Prompt, tools, scoreboard, opened windows list
        for key, value in environment.prompt.items():
            window = text_window(
                value,
                window_id=key,
                interface_prefix="default",
                window_type="segment"
            )
            self.default_windows[window.window_name] = window
        tools_window = text_window(
            self.environment.tools_hint,
            window_id="tools",
            interface_prefix="default",
            window_type="segment"
        )
        self.default_windows[tools_window.window_name] = tools_window
        scoreboard_window = self.scoreboard_manager.get_output_window()
        self.default_windows[scoreboard_window.window_name] = scoreboard_window
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

    def step(self, input: str | AgentForwardContent) -> Dict[str, str]:
        output = self._forward(input)
        tool_call_result = self._post_step(output)
        return tool_call_result

    def add_pre_tool_call_hook(self, hook: Callable[['Agent', str, Optional[str], Optional[str], Optional[str]], Optional[str]]):
        self.pre_tool_call_hooks.append(hook)

    def add_post_tool_call_hook(self, hook: Callable[['Agent', str, str, str, str], Optional[str]]):
        self.post_tool_call_hooks.append(hook)

    def _pre_tool_call_hook(self, model_output: str, context: Optional[str], tool: Optional[str], tool_input: Optional[str]) -> Optional[str|Tuple[str, str, str]]:
        """Hook for tool calling.
        
        If a string is returned, it will be used as the tool call output and
        skips the default tool calling process.
        
        If a tuple is returned, it will be used as the new context-tool-tool_input tuple.
        """
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

    def _post_step(self, model_output: str) -> AgentForwardContent:
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
            context, tool, tool_input = self._parse_llm_output(output)
            # First execute hook.
            hook_output = self._pre_tool_call_hook(output, context, tool, tool_input)
            if isinstance(hook_output, str):
                return hook_output
            elif isinstance(hook_output, tuple):
                context, tool, tool_input = hook_output
                context = context.lower()
                tool = tool.lower()
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
            elif f"text-default-{context}" in self.default_windows:
                ctx = self.default_windows[f"text-default-{context}"]
                tool_output = ctx.invoke(tool, tool_input)
            # Not internal tools. Try context tools.
            elif context in self.toolsets:
                ctx = self.toolsets[context]
                tool_output = ctx.invoke(tool, tool_input, context=context, scoreboard_manager=self.scoreboard_manager)
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

    def _parse_llm_output(self, output: str) -> Tuple[Optional[str], Optional[str], Optional[str | Dict[str, Any]]]:
        return parse_llm_output(output)

    # def interrupt(self, input: str):
    #     self.pause()
    #     self.forward(input)

class HFMixin(StateManagerMixin):
    model: GenerationMixin
    tokenizer: AutoTokenizer
    history_state: Optional[torch.Tensor]
    history_chat: List[Dict[str, str]]
    hf_config: 'Config'

    @dataclass
    class Config:
        GENERATION_CONFIG: GenerationConfig = GenerationConfig()
        MAX_NEW_TOKENS: int = 768
        CHAT_TEMPLATE_ARGS: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, model: GenerationMixin, tokenizer: AutoTokenizer, hf_config: Config, **kwargs):
        self.tokenizer = tokenizer
        self.history_state = None
        self.history_chat = []
        self.model = model
        self.hf_config = hf_config
        super().__init__(**kwargs)

    @staticmethod
    def _debug_output(title: str, content: Any):
        logging.debug(f"=================[{title}]====================")
        s = content if isinstance(content, str) else str(content)
        if len(s) > 2400:
            logging.debug(s[:2400] + "...")
            logging.debug(f"Total output length: {len(s)}")
        else:
            logging.debug(s)

    def _to_chat_format(self, input: str | Dict[str, str]) -> str:
        if isinstance(input, dict):
            chat = [{"role": "assistant", "content": input["ai"]}]
            user = ""
            if "environment" in input:
                user += "Environment: \n" + input["environment"] + "\n\n"
            if "reward" in input:
                user += input["reward"] + "\n\n"
            user += "Now begin your thinking process. Output schema:\n<think>Your thinking process...</think>\n\nOr:\n<think>Your thinking process...</think><tool>{\"context\" : ..., \"tool\" : ..., \"args\" : ...}</tool>"
            chat.append({"role": "user", "content": user})
        else:
            chat = [{"role": "user", "content": input}]
        return chat

    def tokenize(self, input: str | Dict[str, str]) -> torch.Tensor:
        input = self._to_chat_format(input)
        text = self.tokenizer.apply_chat_template(
            input,
            tokenize=False,
            add_generation_prompt=True,
            **self.hf_config.CHAT_TEMPLATE_ARGS
        )
        return input, self.tokenizer([text], return_tensors="pt").input_ids[0]

    def detokenize(self, output: torch.Tensor) -> str:
        return self.tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    def clear_state(self):
        self.history_state = None

    def _forward(self, input: str | Dict[str, str]) -> str:
        chat, tokenized_input = self.tokenize(input)
        self._debug_output("INPUT", chat)
        self.history_chat += chat
        if self.history_state is not None:
            full_input = torch.cat([self.history_state, tokenized_input.to(self.history_state.device)], dim=0)
        else:
            full_input = tokenized_input
        full_input = full_input.unsqueeze(0)
        output_tokens = self.model.generate(
            full_input.to(self.model.device),
            attention_mask=torch.ones_like(full_input).to(self.model.device),
            generation_config=self.hf_config.GENERATION_CONFIG,
            max_new_tokens=self.hf_config.MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id
        )
        self.history_state = output_tokens[0]
        output = output_tokens[0][full_input.shape[1]:]
        output_str = self.detokenize(output)
        self._debug_output("OUTPUT", output_str)
        return output_str

T = TypeVar('T', bound=Any)
class SFTAgent(Agent, Generic[T]):
    """Agent that supports self and automatic SFT training.
    
    Override either `_forward_and_calculate_perplexity` or `_forward`.

    TODO:
    1. raise ValueError if both `_forward_and_calculate_perplexity` and `_forward` are overridden.
    2. raise ValueError if THESHOLD is valid but `_forward_and_calculate_perplexity` is not overridden.
    """
    sft_trainer: SelfSFT
    config: 'Config'
    step_perplexity: Optional[float]

    @dataclass
    class Config(Agent.Config):
        AUTO_SFT: bool = field(default=False, metadata={'description': 'Enable automatic training of the model after each step.'})
        AUTO_SFT_CONFIG: Optional[SelectedSFTConfig] = field(default=None, metadata={'description': 'Configuration for automatic SFT training.'})
        AUTO_SFT_PERPLEXITY_THRESHOLD: float = field(default=0.0, metadata={'description': 'Threshold to stop SFT training. Set to positive value to enable.'})

    def __init__(self, environment: Environment, sft_trainer: SelfSFT, config: Config):
        super().__init__(environment, config)

        if config.AUTO_SFT and config.AUTO_SFT_CONFIG is None:
            raise ValueError("AUTO_SFT_CONFIG is required when AUTO_SFT is enabled.")

        self.sft_trainer = sft_trainer
        self.toolsets["self_sft"] = self.sft_trainer
        self.add_post_tool_call_hook(self._self_sft)
        self.step_perplexity = None

    @staticmethod
    def _self_sft(instance: 'SFTAgent', context: str, tool: str, tool_input: str, tool_output: str):
        if instance.sft_trainer.changed:
            instance.update_model(instance.sft_trainer.get_model())
            instance.sft_trainer.reset_changed()

    def _post_step(self, model_output: str) -> Dict[str, str]:
        result = super()._post_step(model_output)
        if self.config.AUTO_SFT:
            if self.step_perplexity is not None and self.step_perplexity < self.config.AUTO_SFT_PERPLEXITY_THRESHOLD:
                logging.debug(f"Step perplexity is below the threshold. Skipping SFT training.")
                return result
            self.sft_trainer.train(self._to_chat_format(result), self.config.AUTO_SFT_CONFIG)
            self.update_model(self.sft_trainer.get_model())
            self.sft_trainer.reset_changed()
        return result

    def update_model(self, model: T):
        """Update the model of the agent. Default to do nothing."""

    def _to_chat_format(self, input: str | Dict[str, str]) -> str:
        """Convert the input to the chat format. Default to do nothing."""
        return input

    def _forward_and_calculate_perplexity(self, input: str | Dict[str, str]) -> Tuple[str, float]:
        """Low-level forward pass and calculate the perplexity of the output.

        If the agent supports perplexity calculation, it should override this method.
        
        Returns:
            step_output, str: The output of the step.
            step_perplexity, float: The perplexity of the output.
        """
        raise NotImplementedError()

    def _forward(self, input: str | Dict[str, str]) -> str:
        """Default low-level forward pass. If the agent does not support perplexity calculation, it should override this method."""
        step_output, step_perplexity = self._forward_and_calculate_perplexity(input)
        self.step_perplexity = step_perplexity
        return step_output

class SFTHFAgent(HFMixin, SFTAgent[AutoModelForCausalLM]):
    @dataclass
    class Config(SFTAgent.Config, HFMixin.Config):
        pass

    def __init__(self, environment: Environment, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sft_trainer: SelfSFT, config: Config):
        super().__init__(model=model, tokenizer=tokenizer, hf_config=config, environment=environment, sft_trainer=sft_trainer, config=config)

    def _forward(self, input: str | Dict[str, str]) -> str:
        return HFMixin._forward(self, input)