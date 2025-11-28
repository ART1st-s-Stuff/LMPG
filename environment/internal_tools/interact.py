from utils.tool import Toolset
from utils.text import TextWindow, text_window

class Interact(Toolset):
    def __init__(self):
        super().__init__()

    @Toolset.structurized_tool()
    def write(self, content: str) -> TextWindow:
        """Interact tool. Follow the instructions in the prompts for when to use this tool."""
        print(f"Agent: {content}")
        response = input()
        return text_window(response, "response", "interact", "text")