from interact import *

class ShellEnvironment(InteractionProvider):
    def __init__(self):
        self.state = "initial state"

    def interact(self, action: str) -> InteractionResult:
        # Simple interaction logic for demonstration
        self.state += f" -> {action}"
        return InteractionResult(output=self.state, time=1.0)