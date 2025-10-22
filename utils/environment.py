from typing import Sequence, Callable, Optional

from utils.tool import Tool
from utils.scoring import Scoreboard
class Environment:
    def __init__(self, tools: Sequence[Tool], scoreboard: Scoreboard, terminate_predicate: Optional[Callable[[str], bool]] = None):
        self.tools = tools
        self.scoreboard = scoreboard
        self.terminate_predicate = terminate_predicate