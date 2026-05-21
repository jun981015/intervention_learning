from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class ControllerId(IntEnum):
    """Controller identity stored in replay."""

    LEARNER = 0
    EXPERT = 1


class GateReason(IntEnum):
    """Reason code explaining why a gate made its decision."""

    NONE = 0
    RANDOM = 1


@dataclass(frozen=True)
class PolicyOutput:
    """Action proposal returned by a learner or expert policy."""

    action: np.ndarray
    log_prob: float = float("nan")
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateDecision:
    """Controller choice returned by a gating policy."""

    controller_id: ControllerId
    reason: GateReason
    score: float
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def use_expert(self) -> bool:
        """Return whether the selected controller is the expert."""
        return self.controller_id == ControllerId.EXPERT


@dataclass
class StepRecord:
    """One environment step with both policy proposals and the executed action."""

    observation: np.ndarray
    learner: PolicyOutput
    expert: PolicyOutput
    decision: GateDecision
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    next_observation: np.ndarray
    episode_id: int = -1
    episode_step: int = -1
    env_info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Return whether the environment episode ended for any reason."""
        return self.terminated or self.truncated
