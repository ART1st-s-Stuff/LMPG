from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any

from .text import TextWindow, text_window

# 还是有些问题
# 1. 如何安排多个scoreboard？如何处理输出？
# 2. 或许可以设置多种不同score

class Scoreboard(ABC):
    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier
        self.total_points = 0
        self.current_step_points = 0
        self.reasons : List[Tuple[float, str]] = []
    
    def reward(self, points: float, reason: Optional[str] = None):
        self.total_points += points * self.multiplier
        self.current_step_points += points
        self.reasons.append((points, reason))

    def reset_step(self):
        self.current_step_points = 0
        self.reasons.clear()

    @abstractmethod
    def get_current_step_output(self) -> str:
        raise NotImplementedError()
        
class ExplicitScoreboard(Scoreboard):
    def __init__(self, name: str, multiplier: float = 1.0):
        super().__init__(multiplier)
        self.name = name

    @staticmethod
    def format_reasons(reasons: List[Tuple[float, str]]) -> str:
        def format_line(points: float, reason: str) -> str:
            if points > 0:
                return f"  +{points:.2f}: {reason}"
            else:
                return f"  {points:.2f}: {reason}"
        return "\n".join([format_line(points, reason) for points, reason in reasons])

    def get_current_step_output(self) -> str:
        if self.current_step_points == 0:
            ret = f"Scoreboard {self.name}: No updates in this step."
        else:
            ret = f"Scoreboard {self.name}: {self.current_step_points:.2f} points.\nReasons: {self.format_reasons(self.reasons)}\n\n"
        self.reset_step()
        return ret

class ImplicitScoreboard(Scoreboard):
    def get_current_step_output(self) -> str:
        return ""

class ScoreboardManager(ABC):
    """Scoreboard manager that manages multiple scoreboards."""
    def __init__(self):
        self.history : List[str] = []

    @abstractmethod
    def get_scoreboard(self, identifier: Optional[Any] = None) -> Scoreboard:
        raise NotImplementedError()

    @abstractmethod
    def get_current_step_output(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_current_score(self) -> float:
        raise NotImplementedError()

    def get_output_window(self) -> TextWindow:
        return text_window(self.get_current_step_output(),
            window_id="scoreboard",
            interface_prefix="scoreboard",
            window_type="segment"
        )

class DefaultScoreboardManager(ScoreboardManager):
    """Default scoreboard manager with 1 scoreboard."""
    def __init__(self):
        self.default_scoreboard = ExplicitScoreboard(name="default")

    def get_scoreboard(self, identifier: Optional[Any] = None) -> Scoreboard:
        """Get a scoreboard by identifier.
        
        In the default scoreboard manager, we use only 1 scoreboard for all identifiers.
        """
        return self.default_scoreboard

    def get_current_step_output(self) -> str:
        return self.default_scoreboard.get_current_step_output()

    def get_current_score(self) -> float:
        return self.default_scoreboard.total_points