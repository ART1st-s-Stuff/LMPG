from typing import Callable, Dict, List, Literal, TypedDict, Optional

import torch
from torch import nn

from utils.agent import Agent
from train.kl import KLDivergenceEstimator

# Q1 如何让一些critics变得可以从上下文学习？
# 一些critics低学习率，另一些较高；
# 将score条件化
# Q2 学习率是否可以critics相关？
#    或者LM可以自己主动重复来达到增大学习率的效果？

# 多critics的话，或许可以达到类似adaptive LR的效果？



class RLTrainer:
    class Config(TypedDict):
        rl_strategy: Literal["naive", "ppo"]
        kl_divergence_estimator: Optional[KLDivergenceEstimator]
        max_length: int


    def __init__(self,
            actor_model: Agent,
            reference_model: nn.Module,
            critic_models: Dict[str, nn.Module],
            actor_optimizer: torch.optim.Optimizer,
            critic_optimizer: torch.optim.Optimizer,
            get_state: Callable[[List[torch.Tensor], str], torch.Tensor],
            config: Config,
        ):
        self.actor_model = actor_model
        self.reference_model = reference_model
        self.critic_models = critic_models
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.get_state = get_state
        self.config = config
        self.kl_divergence_estimator = config["kl_divergence_estimator"]

        self.mse_loss_fn = nn.MSELoss()

        if self.config["rl_strategy"] == "ppo" and self.kl_divergence_estimator is None:
            raise ValueError("KL divergence estimator is required for PPO")

    def _calculate_value(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Calculate the value of the actor model.

        Note: 
        """
        value = None
        for critic_name, critic_model in self.critic_models.items():
            state = self.get_state(states, critic_name)
            # Do we need a coefficient for each critic?
            if value is None:
                value = critic_model(state)
            else:
                value += critic_model(state)
        return value
    
    def train_critics(self, states: List[torch.Tensor], score: torch.Tensor):
        loss = torch.tensor(0)
        self.critic_optimizer.zero_grad()
        for critic_name, critic_model in self.critic_models.items():
            state = self.get_state(states, critic_name)
            predicted_score = critic_model(state)
            loss += self.mse_loss_fn(predicted_score, score)
        loss.backward()
        self.critic_optimizer.step()
        return loss

    def train_actor(self, inputs: torch.Tensor):
        """
        Args:
            rl_strategy: The reinforcement learning strategy to use.
                - "naive": 
                - "ppo"
        """
        self.actor_optimizer.zero_grad()
        action, states = self.actor_model(inputs)
        value = self._calculate_value(states)
        if self.config["rl_strategy"] == "naive":
            loss = -value.sum()
        elif self.config["rl_strategy"] == "ppo":
            kl = self.kl_divergence_estimator.calculate_approx_kl(self.actor_model)
            loss = -value.sum() + self.config["ppo_config"]["beta"] * kl
        else:
            raise ValueError(f"Invalid RL strategy: {self.config['rl_strategy']}")
        loss.backward()
        self.actor_optimizer.step()
        return loss, value