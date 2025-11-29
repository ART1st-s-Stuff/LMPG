from typing import TypedDict, Dict, Any, Optional
from abc import abstractmethod

from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from utils.tool import Toolset, __ToolsetMeta
from utils.exceptions import InvalidToolArgsException
from utils.scoring import Scoreboard

class SelectedSFTConfig(TypedDict):
    learning_rate: float

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        if len(set(config.keys()) ^ set(SelectedSFTConfig.__annotations__.keys())) != 0:
            return False
        if not isinstance(config["learning_rate"], (int, float)):
            return False
        return True

#T = TypeVar('T', bound=Any)
class SelfSFT(Toolset):
    def __init__(self):
        super().__init__()
        self.changed = False

    @abstractmethod
    def train(self, dataset, config: SelectedSFTConfig) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

    @Toolset.structurized_tool(tool_name="memorize")
    def tool_memorize(self, content: str, config: SelectedSFTConfig) -> None:
        """Memorize the given content.

        Args:
            content, str: The content to memorize. It will be conducted using SFT training.
            config, object: Specify configurations for the SFT training. The modifiable fields are:
                config.learning_rate, float: The learning rate for the SFT training.
        """
        return self.memorize(content, config)
    
    def memorize(self, content: str, config: SelectedSFTConfig) -> None:
        if not SelectedSFTConfig.validate(config):
            raise InvalidToolArgsException(f"Expected argumant config: {{ \"learning_rate\" : \"float: The learning rate for the SFT trainer.\" }}, got {config}")
        self.train([{"text": content}], config)

    def reset_changed(self):
        self.changed = False

class SelfSFT_TRL(SelfSFT):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, sft_config: Dict[str, Any] = {}, peft_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.sft_config = sft_config
        self.peft_config = peft_config
        self.tokenizer = tokenizer
        self.model = model

    def train(self, dataset, config: SelectedSFTConfig) -> None:
        config_dict = self.sft_config
        config_dict.update(config)
        trainer = SFTTrainer(
            model=self.model, args=SFTConfig(**config_dict),
            train_dataset=Dataset.from_list([{"messages":dataset}]),
            processing_class=self.tokenizer,
            peft_config=self.peft_config
        )
        trainer.train()
        self.model = trainer.model
        self.changed = True

    def get_model(self) -> AutoModelForCausalLM:
        return self.model