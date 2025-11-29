from typing import Dict, Callable

from .tool import Toolset
from .scoring import ScoreboardManager

class Environment:
    def __init__(self, tools: Dict[str, Toolset], scoreboard_manager: ScoreboardManager, prompt: Dict[str, str], stop_criteria: Callable[Dict[str, str], bool], max_steps: int = 100):
        self.tools = tools
        self.scoreboard_manager = scoreboard_manager
        self.prompt = prompt
        self.stop_criteria = stop_criteria
        self.num_steps = 0
        self.max_steps = max_steps

    @property
    def tools_hint(self) -> str:
        hint = ""
        for toolset_name, toolset in self.tools.items():
            hint += f"Toolset: {toolset_name}\n"
            hint += f"    {toolset.__doc__}\n"
            for name, prompt, _ in toolset.interface:
                hint += f"    {name}: {prompt}\n"
        return hint

    def run(self, agent):       # Temporary solution
        agent.clear_state()
        for i in range(self.max_steps):
            if i == 0:
                result = agent.step(input=list(self.prompt.values())[0])
            else:
                result = agent.step(result)
            self.num_steps += 1
            if self.stop_criteria(result):
                break
        return result

    def get_avg_score(self) -> float:
        return self.scoreboard_manager.get_current_score() / self.num_steps