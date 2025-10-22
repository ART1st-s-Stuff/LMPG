class Scoreboard:
    def __init__(self):
        self.total_points = 0
        self.current_round_points = 0
        self.reasons = ""
        
    def reward(self, points: float, reason: str = ""):
        self.total_points += points
        self.current_round_points += points
        if points > 0:
            self.reasons += f"+{points}: {reason}\n"
        else:
            self.reasons += f"{points}: {reason}\n"

    def get_current_round_output(self) -> str:
        if self.current_round_points == 0:
            return ""
        return f"Reward: {self.current_round_points}\n{self.reasons}"