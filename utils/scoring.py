class Scoreboard:
    def __init__(self):
        self.points = 0
        
    def reward(self, points: float, reason: str = ""):
        ...