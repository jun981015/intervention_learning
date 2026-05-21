from __future__ import annotations

from typing import Protocol

import numpy as np

from il.utils.types import GateDecision, PolicyOutput


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
    ) -> GateDecision:
        """Choose which policy should control the current environment step."""
        ...
