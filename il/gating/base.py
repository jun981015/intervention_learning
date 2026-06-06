from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from il.utils.types import GateDecision, PolicyOutput


@dataclass(frozen=True)
class GateContext:
    """Optional rollout access for gates that need policy diagnostics."""

    sample_policy: Callable[[str], PolicyOutput]
    policy_observation: Any
    action_dim: int


@runtime_checkable
class ControllerGate(Protocol):
    """Select whether the learner or expert controls the current environment step."""

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
        context: GateContext | None = None,
    ) -> GateDecision:
        """Choose which policy should control the current environment step."""
        ...

    def reset_episode(self) -> None:
        """Clear any gate state that must not cross episode boundaries."""
        ...
