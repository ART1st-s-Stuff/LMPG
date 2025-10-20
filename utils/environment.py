from typing import Sequence
from utils.tool import Tool
from utils.scoring import Scoreboard

class Environment:
    def __init__(self, tools: Sequence[Tool], scoreboard: Scoreboard):
        self.tools = tools
        self.scoreboard = scoreboard