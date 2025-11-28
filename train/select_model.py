from typing import List

from transformers import AutoModelForCausalLM

from utils.agent import SFTAgent
from utils.environment import Environment

def select_model(tasks: List[Environment], agent: SFTAgent, model_pool: List[AutoModelForCausalLM]):
    for task in tasks:
        scores = {}
        for model in model_pool:
            agent.update_model(model)
            agent.set_environment(task)
            task.run(agent)
            scores[model] = task.get_avg_score()
        