from __future__ import annotations

import numpy as np

from il.gating.base import ControllerGate
from il.policies.base import Policy
from il.utils.types import ControllerId, GateDecision, PolicyOutput


def choose_action(
    *,
    step: int,
    observation: np.ndarray,
    learner: Policy,
    expert: Policy,
    gate: ControllerGate,
    learner_rng,
    expert_rng,
    gate_rng: np.random.Generator,
) -> tuple[np.ndarray, PolicyOutput, PolicyOutput, GateDecision]:
    """Sample both policy proposals before gating and return the executed action.

    Sampling both proposals at the same state is intentional: replay needs the
    learner action, expert action, executed action, and gate decision for later
    RL/BC analysis.
    """
    learner_output = learner.sample_action(observation, rng=learner_rng)
    expert_output = expert.sample_action(observation, rng=expert_rng)
    decision = gate.decide(
        step=step,
        observation=observation,
        learner=learner_output,
        expert=expert_output,
        rng=gate_rng,
    )
    action = expert_output.action if decision.controller_id == ControllerId.EXPERT else learner_output.action
    return np.asarray(action, dtype=np.float32), learner_output, expert_output, decision
