from typing import List, Tuple
import random
from copy import deepcopy

from transformers import AutoModelForCausalLM

class WeightManager:
    models: List[AutoModelForCausalLM]

    def __init__(self, model: AutoModelForCausalLM, num_models: int):
        self.model = model
        self.weights = [ deepcopy(model.state_dict()) for _ in range(num_models) ]
        self.scores = [0.0] * num_models
        self.num_models = num_models

    def select_model(self) -> Tuple[AutoModelForCausalLM, int]:
        weight = random.choice(self.weights)
        self.model.load_state_dict(weight)
        return self.model, self.weights.index(weight)

    def set_score(self, index: int, score: float):
        self.scores[index] = score

    def update_model(self):
        best_weight = self.weights[self.scores.index(max(self.scores))]
        self.weights = [ deepcopy(best_weight) for _ in range(self.num_models) ]
        self.scores = [0.0] * self.num_models