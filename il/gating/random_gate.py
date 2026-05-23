from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput


@dataclass
class RandomGate:
    """Bernoulli expert gate used for the first end-to-end pipeline test."""

    expert_probability: float

    def __post_init__(self) -> None:
        """Validate that the expert routing probability is a probability."""
        if not 0.0 <= self.expert_probability <= 1.0:
            raise ValueError("expert_probability must be in [0, 1].")

    def decide(
        self,
        *,
        step: int,
        observation: np.ndarray,
        learner: PolicyOutput,
        expert: PolicyOutput,
        rng: np.random.Generator,
        expert_agent: Any | None = None,
        action_dim: int | None = None,
    ) -> GateDecision:
        """Ignore policy contents and route to expert with fixed probability."""
        del step, observation, learner, expert, expert_agent, action_dim
        sample = float(rng.random())
        use_expert = sample < self.expert_probability
        return GateDecision(
            controller_id=ControllerId.EXPERT if use_expert else ControllerId.LEARNER,
            reason=GateReason.RANDOM,
            score=self.expert_probability,
            info={"random_sample": sample},
        )
