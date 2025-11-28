from typing import TypedDict, Dict, Any, Generic, TypeVar
from abc import abstractmethod

from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from utils.tool import Toolset
from utils.exceptions import InvalidToolInputException
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

T = TypeVar('T', bound=Any)
class SelfSFT(Toolset, Generic[T]):
    def __init__(self):
        super().__init__()
        self.changed = False

    @abstractmethod
    def train(self, dataset, config: SelectedSFTConfig) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_model(self) -> T:
        raise NotImplementedError()

    @Toolset.structurized_tool(tool_name="memorize")
    def tool_memorize(self, content: str, config: SelectedSFTConfig, _scoreboard: Scoreboard) -> None:
        """Memorize the given content.

        Args:
            content, str: The content to memorize. It will be conducted using SFT training.
            config, object: Specify configurations for the SFT training. The modifiable fields are:
                config.learning_rate, float: The learning rate for the SFT training.
        """
        return self.memorize(content, config, _scoreboard)
    
    def memorize(self, content: str, config: SelectedSFTConfig, _scoreboard: Scoreboard) -> None:
        if not SelectedSFTConfig.validate(config):
            raise InvalidToolInputException(input=f"config: {config}", expected="""config.learning_rate, float: The learning rate for the SFT trainer.""")
        self.train([{"text": content}], config)

    def reset_changed(self):
        self.changed = False

class SelfSFT_TRL(SelfSFT):
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: Dict[str, Any] = {}):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def train(self, dataset, config: SelectedSFTConfig) -> None:
        config_dict = self.config
        config_dict.update(config)
        trainer = SFTTrainer(self.model, SFTConfig(**config_dict), train_dataset=Dataset.from_list([{"messages":dataset}]), processing_class=self.tokenizer)
        trainer.train()
        self.model = trainer.model
        self.changed = True

    def get_model(self) -> AutoModelForCausalLM:
        return self.model