from __future__ import annotations

"""Rollout action proposal, gating, and execution-action selection."""

from collections import deque
from typing import Any

import jax
import numpy as np

from il.builders.types import ActorBundle, TrainContext
from il.gating.base import GateContext
from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput


def policy_observation(observation, context: TrainContext):
    """Select the observation view used by current low-dimensional policies."""
    if context.env_spec.state_key is not None and isinstance(observation, dict):
        return observation[context.env_spec.state_key]
    return observation


def residual_policy_observation(policy_obs, base_action: np.ndarray) -> np.ndarray:
    """Concatenate low-dim state and base action for residual actors."""
    if isinstance(policy_obs, dict):
        raise NotImplementedError("residual rollout currently supports low-dim observations only.")
    state = np.asarray(policy_obs, dtype=np.float32).reshape(-1)
    base = np.asarray(base_action, dtype=np.float32).reshape(-1)
    return np.concatenate([state, base], axis=-1).astype(np.float32)


def resolve_residual_scale(context: TrainContext) -> float:
    """Return the residual action scale used consistently by train and eval."""
    return float(context.learner.config.get("residual_scale", context.config["rollout"].get("residual_scale", 1.0)))


def _actor_uses_residual_policy(bundle: ActorBundle | None) -> bool:
    """Return whether an actor expects state-plus-base-action observations."""
    return bool(bundle is not None and bundle.config.get("residual_policy", False))


def _resolve_actor_residual_scale(context: TrainContext, bundle: ActorBundle) -> float:
    """Return the residual scale for a specific residual actor."""
    return float(bundle.config.get("residual_scale", context.config["rollout"].get("residual_scale", 1.0)))


def uses_residual_composition(context: TrainContext) -> bool:
    """Return whether learner actions are composed as base plus residual."""
    rollout_cfg = context.config["rollout"]
    return rollout_cfg.get("action_composition") == "residual" or rollout_cfg.get("execute") == "residual"


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


def _sample_residual_actor_proposal(
    context: TrainContext,
    bundle: ActorBundle,
    policy_obs,
    *,
    base_action: np.ndarray,
    rng,
    reason_if_missing: str,
) -> PolicyOutput:
    """Sample a residual actor and compose it with the supplied base action."""
    base_action = np.asarray(base_action, dtype=np.float32)
    if not np.isfinite(base_action).all():
        raise ValueError(f"{bundle.name} residual actor requires a finite base action.")

    residual_obs = residual_policy_observation(policy_obs, base_action)
    raw_residual_output = _sample_actor(
        bundle,
        residual_obs,
        rng=rng,
        action_dim=context.action_dim,
        reason_if_missing=reason_if_missing,
    )
    if np.isnan(raw_residual_output.action).any():
        raise ValueError(f"{bundle.name} residual actor produced NaN action.")

    residual_scale = _resolve_actor_residual_scale(context, bundle)
    raw_residual = np.asarray(raw_residual_output.action, dtype=np.float32)
    residual_action = raw_residual * residual_scale
    composed_action = np.clip(base_action + residual_action, -1.0, 1.0).astype(np.float32)
    return PolicyOutput(
        action=composed_action,
        log_prob=raw_residual_output.log_prob,
        info={
            **raw_residual_output.info,
            "base_action": base_action,
            "raw_residual_action": raw_residual,
            "residual_action": residual_action,
            "residual_scale": residual_scale,
            "composed_action": composed_action,
            "base_kind": context.base.kind if context.base is not None else "",
            "base_checkpoint_path": str(context.base.checkpoint_path) if context.base and context.base.checkpoint_path else "",
        },
    )


def _sample_expert_proposal(
    context: TrainContext,
    policy_obs,
    *,
    rng,
    base_action: np.ndarray | None,
) -> PolicyOutput:
    """Sample an expert proposal, including residual experts that need base actions."""
    if _actor_uses_residual_policy(context.expert):
        if context.expert is None:
            return _missing_policy_output(context.action_dim, reason="expert_not_sampled")
        if base_action is None:
            raise ValueError("residual expert requires residual action composition and an available base action.")
        return _sample_residual_actor_proposal(
            context,
            context.expert,
            policy_obs,
            base_action=base_action,
            rng=rng,
            reason_if_missing="expert_not_sampled",
        )
    return _sample_actor(
        context.expert,
        policy_obs,
        rng=rng,
        action_dim=context.action_dim,
        reason_if_missing="expert_not_sampled",
    )


def _rollout_state(context: TrainContext) -> dict[str, Any]:
    """Return mutable rollout state, creating it for older contexts if needed."""
    if not hasattr(context, "rollout_state") or context.rollout_state is None:
        context.rollout_state = {}
    return context.rollout_state


def reset_rollout_state(context: TrainContext, *, reset_gate: bool = False) -> None:
    """Clear per-episode rollout caches, and optionally gate state."""
    _rollout_state(context).clear()
    if reset_gate and context.gate is not None:
        context.gate.reset_episode()


def _base_action_queue(context: TrainContext):
    """Return the base policy action queue used by residual rollout."""
    state = _rollout_state(context)
    queue = state.get("base_action_queue")
    if queue is None:
        queue = deque()
        state["base_action_queue"] = queue
    return queue


def _enqueue_base_policy_output(context: TrainContext, output: PolicyOutput) -> None:
    """Store a base policy chunk as primitive actions to execute one-by-one."""
    queue = _base_action_queue(context)
    chunk = output.info.get("full_action_chunk")
    if chunk is None:
        flat_action = np.asarray(output.action, dtype=np.float32).reshape(-1)
        if flat_action.size % context.action_dim != 0:
            raise ValueError(f"base action dim mismatch: expected multiple of {context.action_dim}, got {flat_action.size}.")
        actions = flat_action.reshape(-1, context.action_dim)
    else:
        actions = np.asarray(chunk, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"base full_action_chunk must be [horizon, action_dim], got shape {actions.shape}.")
    if actions.shape[1] != context.action_dim:
        raise ValueError(f"base action dim mismatch: expected {context.action_dim}, got shape {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError("residual rollout requires finite base action chunk values.")

    chunk_size = int(actions.shape[0])
    for idx, action in enumerate(actions):
        info = {key: value for key, value in output.info.items() if key != "full_action_chunk"}
        info.update(
            {
                "base_chunk_index": idx,
                "base_chunk_size": chunk_size,
                "base_queue_refill": int(idx == 0),
                "base_queue_remaining_after_pop": chunk_size - idx - 1,
            }
        )
        queue.append(
            PolicyOutput(
                action=np.asarray(action, dtype=np.float32),
                log_prob=output.log_prob,
                info=info,
            )
        )


def _sample_or_pop_base_action(context: TrainContext, observation, *, rng) -> PolicyOutput:
    """Return the next base action, querying the base policy only when the queue is empty."""
    queue = _base_action_queue(context)
    if not queue:
        policy_obs = policy_observation(observation, context)
        output = _sample_actor(
            context.base,
            policy_obs,
            rng=rng,
            action_dim=context.action_dim,
            reason_if_missing="base_not_sampled",
        )
        _enqueue_base_policy_output(context, output)
    output = queue.popleft()
    if np.isnan(output.action).any():
        raise ValueError("residual rollout requires a finite base action.")
    return output


def sample_base_action(context: TrainContext, observation, *, rng) -> PolicyOutput:
    """Return the current base action for residual rollout."""
    state = _rollout_state(context)
    pending = state.pop("pending_base_output", None)
    if pending is not None:
        if np.isnan(pending.action).any():
            raise ValueError("residual rollout requires a finite pending base action.")
        return pending
    return _sample_or_pop_base_action(context, observation, rng=rng)


def prepare_next_base_action(context: TrainContext, observation, *, rng) -> PolicyOutput:
    """Precompute the next state's base action and keep it for the next rollout step."""
    state = _rollout_state(context)
    if "pending_base_output" in state:
        raise ValueError("pending base action already exists; sample it before preparing another next_base_action.")
    output = _sample_or_pop_base_action(context, observation, rng=rng)
    state["pending_base_output"] = output
    return output


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
    observation,
    policy_obs,
    *,
    rollout_cfg: dict[str, Any],
    execute: str,
    step: int,
) -> tuple[PolicyOutput, PolicyOutput]:
    """Sample learner and expert proposals for the current state."""
    context.rng, learner_rng, expert_rng = jax.random.split(context.rng, 3)
    sample_learner = bool(rollout_cfg.get("sample_learner", True)) or execute in ("learner", "gate")
    sample_expert = bool(rollout_cfg.get("sample_expert", False)) or execute in ("expert", "gate")

    base_action_for_expert = None
    if not sample_learner:
        learner_output = _missing_policy_output(context.action_dim, reason="learner_not_sampled")
    elif uses_residual_composition(context):
        learner_rng, base_rng, residual_rng = jax.random.split(learner_rng, 3)
        learner_output = _sample_residual_learner_proposal(
            context,
            observation,
            policy_obs,
            step=step,
            base_rng=base_rng,
            residual_rng=residual_rng,
        )
        base_action_for_expert = learner_output.info.get("base_action")
    else:
        learner_output = _sample_actor(
            context.learner,
            policy_obs,
            rng=learner_rng,
            action_dim=context.action_dim,
            reason_if_missing="learner_not_sampled",
        )
    expert_output = (
        _sample_expert_proposal(
            context,
            policy_obs,
            rng=expert_rng,
            base_action=base_action_for_expert,
        )
        if sample_expert
        else _missing_policy_output(context.action_dim, reason="expert_not_sampled")
    )
    return learner_output, expert_output


def _sample_fresh_residual_learner_proposal(
    context: TrainContext,
    policy_obs,
    *,
    step: int,
    base_rng,
    residual_rng,
) -> PolicyOutput:
    """Sample a residual-composed learner proposal without consuming the base queue."""
    base_output = _sample_actor(
        context.base,
        policy_obs,
        rng=base_rng,
        action_dim=context.action_dim,
        reason_if_missing="base_not_sampled",
    )
    if not np.isfinite(np.asarray(base_output.action, dtype=np.float32)).all():
        raise ValueError("residual uncertainty sampling requires a finite base action.")
    return _compose_residual_learner_proposal(
        context,
        policy_obs,
        step=step,
        base_output=base_output,
        residual_rng=residual_rng,
    )


def _sample_gate_policy_source(
    context: TrainContext,
    source: str,
    policy_obs,
    *,
    step: int,
    rng,
    learner_output: PolicyOutput,
) -> PolicyOutput:
    """Sample one gate diagnostic policy source without advancing rollout queues."""
    if source == "learner":
        if uses_residual_composition(context):
            base_rng, residual_rng = jax.random.split(rng)
            return _sample_fresh_residual_learner_proposal(
                context,
                policy_obs,
                step=step,
                base_rng=base_rng,
                residual_rng=residual_rng,
            )
        return _sample_actor(
            context.learner,
            policy_obs,
            rng=rng,
            action_dim=context.action_dim,
            reason_if_missing="learner_not_sampled",
        )
    if source == "expert":
        return _sample_expert_proposal(
            context,
            policy_obs,
            rng=rng,
            base_action=learner_output.info.get("base_action"),
        )
    if source == "base":
        return _sample_actor(
            context.base,
            policy_obs,
            rng=rng,
            action_dim=context.action_dim,
            reason_if_missing="base_not_sampled",
        )
    raise ValueError(f"Unsupported gate policy source: {source!r}")


def _make_gate_context(
    context: TrainContext,
    policy_obs,
    *,
    step: int,
    learner_output: PolicyOutput,
) -> GateContext:
    """Build optional rollout access for gates that compute policy diagnostics."""

    def sample_policy(source: str) -> PolicyOutput:
        context.rng, sample_rng = jax.random.split(context.rng)
        return _sample_gate_policy_source(
            context,
            source,
            policy_obs,
            step=step,
            rng=sample_rng,
            learner_output=learner_output,
        )

    return GateContext(
        sample_policy=sample_policy,
        policy_observation=policy_obs,
        action_dim=context.action_dim,
    )


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
        gate_context = _make_gate_context(
            context,
            policy_obs,
            step=step,
            learner_output=learner_output,
        )
        return context.gate.decide(
            step=step,
            observation=policy_obs,
            learner=learner_output,
            expert=expert_output,
            rng=context.gate_rng,
            expert_agent=context.expert.agent if context.expert is not None else None,
            action_dim=context.action_dim,
            context=gate_context,
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


def _residual_warmup_output(context: TrainContext, base_action: np.ndarray, *, rng, residual_scale: float, step: int) -> PolicyOutput | None:
    """Return a random residual warmup action, or None when warmup is inactive."""
    rollout_cfg = context.config["rollout"]
    warmup_steps = int(rollout_cfg.get("residual_warmup_steps", 0))
    if warmup_steps <= 0 or step > warmup_steps:
        return None

    noise_scale = float(rollout_cfg.get("warmup_noise_scale", rollout_cfg.get("random_action_noise_scale", 0.2)))
    use_base_policy = bool(rollout_cfg.get("use_base_policy_for_warmup", True))
    noise = np.asarray(
        jax.random.uniform(rng, (context.action_dim,), minval=-noise_scale, maxval=noise_scale),
        dtype=np.float32,
    )
    if use_base_policy:
        residual_action = noise
    else:
        residual_action = noise - np.asarray(base_action, dtype=np.float32)
    raw_residual = residual_action / max(residual_scale, 1e-6)
    return PolicyOutput(
        action=residual_action.astype(np.float32),
        log_prob=float("nan"),
        info={
            "raw_residual_action": raw_residual.astype(np.float32),
            "residual_action": residual_action.astype(np.float32),
            "residual_warmup": 1,
            "residual_warmup_steps": warmup_steps,
            "warmup_noise_scale": noise_scale,
            "use_base_policy_for_warmup": int(use_base_policy),
        },
    )



def _compose_residual_learner_proposal(
    context: TrainContext,
    policy_obs,
    *,
    step: int,
    base_output: PolicyOutput,
    residual_rng,
) -> PolicyOutput:
    """Compose a base action and residual learner output into an executable action."""
    residual_scale = resolve_residual_scale(context)
    base_action = np.asarray(base_output.action, dtype=np.float32)

    raw_residual_output = _residual_warmup_output(
        context,
        base_action,
        rng=residual_rng,
        residual_scale=residual_scale,
        step=step,
    )
    if raw_residual_output is None:
        residual_obs = residual_policy_observation(policy_obs, base_action)
        raw_residual_output = _sample_actor(
            context.learner,
            residual_obs,
            rng=residual_rng,
            action_dim=context.action_dim,
            reason_if_missing="residual_learner_not_sampled",
        )
        if np.isnan(raw_residual_output.action).any():
            raise ValueError("residual learner produced NaN action.")
        raw_residual = np.asarray(raw_residual_output.action, dtype=np.float32)
        residual_action = raw_residual * residual_scale
    else:
        residual_action = np.asarray(raw_residual_output.action, dtype=np.float32)
        raw_residual = np.asarray(raw_residual_output.info["raw_residual_action"], dtype=np.float32)

    learner_action = np.clip(base_action + residual_action, -1.0, 1.0).astype(np.float32)
    return PolicyOutput(
        action=learner_action,
        log_prob=raw_residual_output.log_prob,
        info={
            **raw_residual_output.info,
            "base_action": base_action,
            "raw_residual_action": raw_residual,
            "residual_action": residual_action,
            "residual_scale": residual_scale,
            "learner_action": learner_action,
            "base_kind": context.base.kind,
            "base_checkpoint_path": str(context.base.checkpoint_path) if context.base.checkpoint_path else "",
        },
    )


def _sample_residual_learner_proposal(
    context: TrainContext,
    observation,
    policy_obs,
    *,
    step: int,
    base_rng,
    residual_rng,
) -> PolicyOutput:
    """Return the learner's full residual-composed action proposal."""
    if context.base is None:
        raise ValueError("residual action composition requires a built base actor.")

    base_output = sample_base_action(context, observation, rng=base_rng)
    return _compose_residual_learner_proposal(
        context,
        policy_obs,
        step=step,
        base_output=base_output,
        residual_rng=residual_rng,
    )


def _choose_residual_action(context: TrainContext, observation, *, step: int):
    """Sample base and residual policies, then execute their clipped sum."""
    policy_obs = policy_observation(observation, context)
    context.rng, base_rng, residual_rng, expert_rng = jax.random.split(context.rng, 4)

    learner_output = _sample_residual_learner_proposal(
        context,
        observation,
        policy_obs,
        step=step,
        base_rng=base_rng,
        residual_rng=residual_rng,
    )
    residual_scale = learner_output.info["residual_scale"]
    action = learner_output.action

    expert_output = (
        _sample_expert_proposal(
            context,
            policy_obs,
            rng=expert_rng,
            base_action=learner_output.info.get("base_action"),
        )
        if bool(context.config["rollout"].get("sample_expert", False))
        else _missing_policy_output(context.action_dim, reason="expert_not_sampled")
    )
    decision = GateDecision(
        controller_id=ControllerId.LEARNER,
        reason=GateReason.NONE,
        score=0.0,
        info={"execute": "residual", "residual_scale": residual_scale},
    )
    return action, learner_output, expert_output, decision


def choose_rollout_action(context: TrainContext, observation, *, step: int):
    """Run policy proposal sampling, gate decision, and action selection."""
    rollout_cfg = context.config["rollout"]
    execute = rollout_cfg.get("execute", "learner")
    if execute == "residual":
        return _choose_residual_action(context, observation, step=step)

    policy_obs = policy_observation(observation, context)
    learner_output, expert_output = _sample_action_proposals(
        context,
        observation,
        policy_obs,
        rollout_cfg=rollout_cfg,
        execute=execute,
        step=step,
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
