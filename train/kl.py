from abc import ABC, abstractmethod
import copy
from typing import TypedDict, Literal, Optional, Dict

import torch
from torch import nn

from utils.agent import Agent

class KLRolloutSampler(ABC):
    @property
    @abstractmethod
    def static(self) -> bool:
        """Whether the rollout sampler is static."""
        raise NotImplementedError()

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """Samples from the model.

        Args:
            sample_sources: Prompts for sampling. A batch.
            rollout_config: The rollout config.

        Returns:
            The rollout results.
        """
        raise NotImplementedError()

class KLDivergenceEstimator:
    class Config(TypedDict):
        """
        comparison_type:
            - "reference": Compare with the reference model.
            - "last_step": Compare with the model from last step.
            - "non_memory": Only applicable if the original model supports progressive learning. Use the non-memory part as the reference model.
        """
        kl_estimator: Literal["k1", "k2", "k3"]
        comparison_type: Literal["reference", "last_step", "non_memory"]
        rollout_config: Dict

    def __init__(self, config: Config, sampler: KLRolloutSampler, reference_model: Optional[nn.Module] = None):
        self.config = config
        self.reference_model = reference_model
        self.sampler = sampler
        if sampler.static and config["comparison_type"] == "reference":
            self.static = True
            self.static_samples = sampler.sample()
            self.static_logits_ref = self._rollout(reference_model, self.static_samples, config["rollout_config"])
        else:
            self.static = False

        if config["comparison_type"] == "non_memory":
            raise NotImplementedError("Non-memory comparison is not implemented yet")

    @torch.no_grad()
    @staticmethod
    def _rollout(
            model: Agent,
            sample_inputs: torch.Tensor,
            rollout_config: Dict,
            ):
        """Generate rollout results to compute KL divergence.

        Args:
            model: The model to rollout.
            sample_sources: Prompts for sampling. A batch.
            rollout_config: The rollout config.

        Returns:
            The rollout results.
        """
        outputs, logits = model._forward(sample_inputs, rollout_config)
        return logits

    def _calculate_approx_kl(self, logits: torch.Tensor, logits_ref: torch.Tensor) -> torch.Tensor:
        """Calculate the approximate KL divergence between the logits.

        Args:
            logits: The logits of the model.
            logits_ref: The logits of the reference model.

        Returns:
            The approximate KL divergence.
        """
        log_ratio = logits.float() - logits_ref.float()
        match self.config["kl_estimator"]:
            case "k1":
                # Do nothing
                pass
            case "k2":
                log_ratio = log_ratio**2 / 2.0
            case "k3":
                log_ratio = -log_ratio
                log_ratio = log_ratio.exp() - 1 - log_ratio
        return log_ratio

    def calculate_approx_kl(self, model: Agent) -> torch.Tensor:
        """Calculate the approximate KL divergence between the logits of the model and the reference model.
        
        Automatically samples from the model and the reference model.

        Args:
            model: The model to calculate the KL divergence for.

        Returns:
            The approximate KL divergence.
        """
        logits = self._rollout(model, self.sampler.sample(), self.config["rollout_config"])
        if self.static:
            logits_ref = self.static_logits_ref
        else:
            if self.reference_model is None:
                logits_ref = torch.zeros_like(logits)
            else:
                logits_ref = self._rollout(self.reference_model, self.sampler.sample(), self.config["rollout_config"])
            if self.config["comparison_type"] == "reference":
                self.reference_model = copy.deepcopy(model.model)
                # TODO: Clear gradients
                # self.reference_model.requires_grad_(False)
        return self._calculate_approx_kl(logits, logits_ref)