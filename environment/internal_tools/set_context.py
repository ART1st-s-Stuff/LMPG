from utils.tool import Toolset

class SetContextTool(Toolset):
    context_prompt: str = ""

    @Toolset.structurized_tool()
    def set_context(self, context: str) -> None:
        """Memorize the given content.

        Args:
            content, str: The content to memorize. It will be conducted using SFT training.
            config, object: Specify configurations for the SFT training. The modifiable fields are:
                config.learning_rate, float: The learning rate for the SFT training.
        """
        self.context_prompt = context