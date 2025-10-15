from abc import ABC, abstractmethod

class Tool(ABC):
    def __init__(self, scoreboard):
        self.scoreboard = scoreboard
        
        # TODO: Tool call score