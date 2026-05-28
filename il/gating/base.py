from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from il.utils.types import GateDecision, PolicyOutput


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
    ) -> GateDecision:
        """Choose which policy should control the current environment step."""
        ...

    def reset_episode(self) -> None:
        """Clear any gate state that must not cross episode boundaries."""
        ...
