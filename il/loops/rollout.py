from __future__ import annotations

"""Rollout action proposal, gating, and execution-action selection."""

from typing import Any

import jax
import numpy as np

from il.builders.types import ActorBundle, TrainContext
from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput


def policy_observation(observation, context: TrainContext):
    """Select the observation view used by current low-dimensional policies."""
    if context.env_spec.state_key is not None and isinstance(observation, dict):
        return observation[context.env_spec.state_key]
    return observation


def _missing_policy_output(action_dim: int, *, reason: str) -> PolicyOutput:
    """Return a NaN action placeholder when a proposal is intentionally absent."""
    return PolicyOutput(
        action=np.full(action_dim, np.nan, dtype=np.float32),
        log_prob=float("nan"),
        info={"missing": reason},
    )


def _sample_actor(
    bundle: ActorBundle | None,
    observation,
    *,
    rng,
    action_dim: int,
    reason_if_missing: str,
) -> PolicyOutput:
    """Sample a policy proposal or return a placeholder."""
    if bundle is None or bundle.policy is None:
        return _missing_policy_output(action_dim, reason=reason_if_missing)
    return bundle.policy.sample_action(observation, rng=rng)


def _fixed_decision(controller_id: ControllerId, *, reason: str) -> GateDecision:
    """Create a non-gated controller decision."""
    return GateDecision(
        controller_id=controller_id,
        reason=GateReason.NONE,
        score=0.0,
        info={"execute": reason},
    )


def _sample_action_proposals(
    context: TrainContext,
    policy_obs,
    *,
    rollout_cfg: dict[str, Any],
    execute: str,
) -> tuple[PolicyOutput, PolicyOutput]:
    """Sample learner and expert proposals for the current state."""
    context.rng, learner_rng, expert_rng = jax.random.split(context.rng, 3)
    sample_learner = bool(rollout_cfg.get("sample_learner", True)) or execute in ("learner", "gate")
    sample_expert = bool(rollout_cfg.get("sample_expert", False)) or execute in ("expert", "gate")

    learner_output = (
        _sample_actor(
            context.learner,
            policy_obs,
            rng=learner_rng,
            action_dim=context.action_dim,
            reason_if_missing="learner_not_sampled",
        )
        if sample_learner
        else _missing_policy_output(context.action_dim, reason="learner_not_sampled")
    )
    expert_output = (
        _sample_actor(
            context.expert,
            policy_obs,
            rng=expert_rng,
            action_dim=context.action_dim,
            reason_if_missing="expert_not_sampled",
        )
        if sample_expert
        else _missing_policy_output(context.action_dim, reason="expert_not_sampled")
    )
    return learner_output, expert_output


def _decide_controller(
    context: TrainContext,
    policy_obs,
    *,
    step: int,
    execute: str,
    learner_output: PolicyOutput,
    expert_output: PolicyOutput,
) -> GateDecision:
    """Choose which proposal should control the environment step."""
    if execute == "gate":
        if context.gate is None:
            raise ValueError("rollout.execute='gate' requires a configured gate.")
        return context.gate.decide(
            step=step,
            observation=policy_obs,
            learner=learner_output,
            expert=expert_output,
            rng=context.gate_rng,
            expert_agent=context.expert.agent if context.expert is not None else None,
            action_dim=context.action_dim,
        )
    if execute == "learner":
        return _fixed_decision(ControllerId.LEARNER, reason="learner")
    if execute == "expert":
        if context.expert is None:
            raise ValueError("rollout.execute='expert' requires an expert actor.")
        return _fixed_decision(ControllerId.EXPERT, reason="expert")
    raise ValueError(f"Unsupported rollout.execute: {execute!r}")


def _select_executed_action(
    decision: GateDecision,
    *,
    learner_output: PolicyOutput,
    expert_output: PolicyOutput,
) -> np.ndarray:
    """Return the action selected by the controller decision."""
    action = expert_output.action if decision.controller_id == ControllerId.EXPERT else learner_output.action
    if np.isnan(action).any():
        raise ValueError(f"Selected controller produced NaN action: controller={decision.controller_id}")
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def choose_rollout_action(context: TrainContext, observation, *, step: int):
    """Run policy proposal sampling, gate decision, and action selection."""
    rollout_cfg = context.config["rollout"]
    execute = rollout_cfg.get("execute", "learner")
    policy_obs = policy_observation(observation, context)

    learner_output, expert_output = _sample_action_proposals(
        context,
        policy_obs,
        rollout_cfg=rollout_cfg,
        execute=execute,
    )
    decision = _decide_controller(
        context,
        policy_obs,
        step=step,
        execute=execute,
        learner_output=learner_output,
        expert_output=expert_output,
    )
    action = _select_executed_action(
        decision,
        learner_output=learner_output,
        expert_output=expert_output,
    )
    return action, learner_output, expert_output, decision
