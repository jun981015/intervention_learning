from __future__ import annotations

"""Recipe-driven online training loop.

This module owns the env-step loop. Builders construct components, algorithms
own gradient updates, and buffers own storage/sampling.
"""

import time
from typing import Any

import jax
import numpy as np

from il.buffers.routing import route_episode_to_buffers
from il.buffers.schema import step_record_to_transition
from il.buffers.mixed import MixedReplaySampler, MixedSamplingSpec
from il.builders.types import ActorBundle, TrainContext
from il.logging import MetricLogger
from il.utils.config import TrainingConfig
from il.utils.flax_utils import save_agent
from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput, StepRecord


def tree_to_float_dict(tree: dict) -> dict[str, float]:
    """Convert scalar metric leaves to plain Python floats."""
    out = {}
    for key, value in tree.items():
        arr = np.asarray(value)
        if arr.size == 1:
            out[str(key)] = float(arr.reshape(()))
    return out


def _policy_observation(observation, context: TrainContext):
    """Select the low-dimensional state view used by current policies."""
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


def _choose_rollout_action(context: TrainContext, observation, *, step: int):
    """Sample learner/expert proposals and choose the action to execute."""
    rollout_cfg = context.config["rollout"]
    execute = rollout_cfg.get("execute", "learner")
    policy_obs = _policy_observation(observation, context)

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

    if execute == "gate":
        if context.gate is None:
            raise ValueError("rollout.execute='gate' requires a configured gate.")
        decision = context.gate.decide(
            step=step,
            observation=policy_obs,
            learner=learner_output,
            expert=expert_output,
            rng=context.gate_rng,
        )
    elif execute == "learner":
        decision = _fixed_decision(ControllerId.LEARNER, reason="learner")
    elif execute == "expert":
        if context.expert is None:
            raise ValueError("rollout.execute='expert' requires an expert actor.")
        decision = _fixed_decision(ControllerId.EXPERT, reason="expert")
    else:
        raise ValueError(f"Unsupported rollout.execute: {execute!r}")

    action = expert_output.action if decision.controller_id == ControllerId.EXPERT else learner_output.action
    if np.isnan(action).any():
        raise ValueError(f"Selected controller produced NaN action: controller={decision.controller_id}")
    return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0), learner_output, expert_output, decision


def _select_update_source(context: TrainContext, spec: dict[str, Any], train_cfg: TrainingConfig):
    """Return the replay source requested by an update spec."""
    if train_cfg.sampling_fractions is not None:
        return MixedReplaySampler(context.buffers, MixedSamplingSpec(train_cfg.sampling_fractions))
    return context.buffers.get(spec.get("source", "online"))


def _make_update_config(context: TrainContext, target: ActorBundle, spec: dict[str, Any]) -> TrainingConfig:
    """Resolve update sampling knobs from recipe + target actor config."""
    return TrainingConfig(
        batch_size=int(spec.get("batch_size", context.config["train"]["batch_size"])),
        utd_ratio=int(spec.get("utd_ratio", spec.get("utd", 1))),
        horizon_length=int(spec.get("horizon_length", target.config.get("horizon_length", 1))),
        discount=float(spec.get("discount", target.config.get("discount", 0.99))),
        sampling_fractions=spec.get("sampling_fractions"),
    )


def _sample_update_batch(source, train_cfg: TrainingConfig) -> dict:
    """Sample a sequence batch and stack it for optional UTD updates."""
    batch = source.sample_sequence(
        train_cfg.batch_size * train_cfg.utd_ratio,
        sequence_length=train_cfg.horizon_length,
        discount=train_cfg.discount,
    )
    if train_cfg.utd_ratio == 1:
        return batch
    return jax.tree_util.tree_map(
        lambda x: x.reshape((train_cfg.utd_ratio, train_cfg.batch_size) + x.shape[1:]),
        batch,
    )


def _target_bundle(context: TrainContext, name: str) -> ActorBundle:
    """Return the actor bundle named by an update spec."""
    if name == "learner":
        return context.learner
    if name == "expert" and context.expert is not None:
        return context.expert
    raise KeyError(f"Unknown update target {name!r}.")


def _set_bundle_agent(bundle: ActorBundle, agent) -> None:
    """Keep the trainable agent and its policy view synchronized."""
    bundle.agent = agent
    if bundle.policy is not None and hasattr(bundle.policy, "agent"):
        bundle.policy.agent = agent


def _assert_finite_bc_targets(batch: dict, key: str, *, update_name: str) -> None:
    """Fail fast if a BC target contains missing expert labels."""
    if key not in batch:
        raise KeyError(f"BC target_action_key {key!r} is missing from sampled batch for {update_name!r}.")
    targets = np.asarray(batch[key])
    finite = np.isfinite(targets)
    if bool(finite.all()):
        return
    bad_count = int(targets.size - finite.sum())
    raise ValueError(
        f"BC target_action_key {key!r} for update {update_name!r} contains "
        f"{bad_count}/{targets.size} NaN or Inf values. This usually means expert actions "
        "were not queried/stored before DAgger relabel training."
    )


def _prepare_bc_batch(target: ActorBundle, spec: dict[str, Any], batch: dict) -> dict:
    """Alias requested BC labels to the target actor's configured label key and validate them."""
    requested_key = spec.get("target_action_key")
    configured_key = target.config.get("target_action_key")
    update_name = spec.get("name") or target.name
    target_key = requested_key or configured_key
    if target_key is not None:
        _assert_finite_bc_targets(batch, target_key, update_name=update_name)
    if requested_key and configured_key and requested_key != configured_key:
        batch = dict(batch)
        batch[configured_key] = batch[requested_key]
    return batch


def run_update_spec(context: TrainContext, spec: dict[str, Any]) -> dict[str, float]:
    """Run one configured learner update and return scalar metrics."""
    target = _target_bundle(context, spec.get("target", "learner"))
    if target.agent is None:
        raise ValueError(f"Update target {target.name!r} has no trainable agent.")

    train_cfg = _make_update_config(context, target, spec)
    source = _select_update_source(context, spec, train_cfg)
    batch = _sample_update_batch(source, train_cfg)
    objective = spec.get("objective", "bc")
    if objective == "bc":
        batch = _prepare_bc_batch(target, spec, batch)
    elif objective != "rl":
        raise ValueError(f"Unsupported update objective: {objective!r}")

    if train_cfg.utd_ratio > 1:
        agent, info = target.agent.batch_update(batch)
    else:
        agent, info = target.agent.update(batch)
    _set_bundle_agent(target, agent)
    prefix = spec.get("name") or f"{target.name}_{objective}"
    return {f"{prefix}/{key}": value for key, value in tree_to_float_dict(info).items()}


def _make_logger(context: TrainContext) -> MetricLogger:
    """Create the metric logger configured for this run."""
    run_cfg = context.config["run"]
    train_cfg = context.config["train"]
    return MetricLogger(
        run_dir=context.paths.run_dir,
        config=context.config,
        stdout_interval=int(train_cfg.get("log_interval", 0)),
        jsonl_enabled=bool(run_cfg.get("jsonl", True)),
        csv_enabled=bool(run_cfg.get("csv", True)),
        wandb_enabled=bool(run_cfg.get("wandb", False)),
    )


def _evaluate_policy(context: TrainContext, *, step: int) -> dict[str, float]:
    """Run simple learner-only evaluation."""
    eval_env = context.eval_env
    if eval_env is None or context.learner.policy is None:
        return {}
    train_cfg = context.config["train"]
    episodes = int(train_cfg.get("eval_episodes", 0))
    if episodes <= 0:
        return {}

    returns = []
    lengths = []
    successes = []
    seed = int(context.config["run"]["seed"]) + step
    rng = context.rng
    for episode_idx in range(episodes):
        observation, _ = eval_env.reset(options={"seed": seed + episode_idx})
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_success = 0.0
        while not done:
            rng, action_rng = jax.random.split(rng)
            policy_obs = _policy_observation(observation, context)
            output = context.learner.policy.sample_action(policy_obs, rng=action_rng)
            action = np.clip(np.asarray(output.action, dtype=np.float32), -1.0, 1.0)
            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_length += 1
            episode_success = max(episode_success, float(info.get("success", 0.0)))
        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(episode_success)
    context.rng = rng
    return {
        "eval/return": float(np.mean(returns)),
        "eval/length": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(successes)),
    }


def _save_train_state(context: TrainContext, step: int) -> None:
    """Save trainable actor checkpoints."""
    if context.learner.agent is not None and context.learner.train:
        save_agent(context.learner.agent, context.paths.run_dir, step)
    if context.expert is not None and context.expert.agent is not None and context.expert.train:
        expert_dir = context.paths.run_dir / "expert"
        expert_dir.mkdir(parents=True, exist_ok=True)
        save_agent(context.expert.agent, expert_dir, step)


def _save_buffers(context: TrainContext) -> None:
    """Persist all physical replay buffers."""
    for name, buffer in context.buffers.as_dict().items():
        buffer.save_npz(context.paths.run_dir / f"{name}_replay_buffer.npz")


def run_train_loop(context: TrainContext) -> TrainContext:
    """Run the configured online training loop."""
    train_cfg = context.config["train"]
    replay_cfg = context.config["replay"]
    steps = int(train_cfg["steps"])
    start_training = int(train_cfg["start_training"])
    log_interval = int(train_cfg["log_interval"])
    eval_interval = int(train_cfg.get("eval_interval", 0))
    save_interval = int(train_cfg.get("save_interval", 0))
    include_failed_interventions = bool(replay_cfg.get("include_failed_interventions", False))
    demo_insert_mode = replay_cfg.get("demo_insert_mode", "append")

    logger = _make_logger(context)
    observation, _ = context.env.reset(options={"seed": int(context.config["run"]["seed"])})
    episode: list[dict] = []
    episode_return = 0.0
    episode_length = 0
    episode_success = 0.0
    episode_count = 0
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    recent_successes: list[float] = []
    route_metrics = {"demo_added": 0, "demo_removed": 0, "demo_skipped": 0, "intervention_added": 0}

    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    print(f"[train] starting loop steps={steps} run_dir={context.paths.run_dir}", flush=True)

    for step in range(1, steps + 1):
        action, learner_output, expert_output, decision = _choose_rollout_action(
            context,
            observation,
            step=step,
        )
        next_observation, reward, terminated, truncated, info = context.env.step(action)
        transition = step_record_to_transition(
            StepRecord(
                observation=observation,
                learner=learner_output,
                expert=expert_output,
                decision=decision,
                action=action,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                next_observation=next_observation,
                episode_id=episode_count,
                episode_step=episode_length,
                env_info=info,
            )
        )
        context.buffers.online.add_transition(transition)
        episode_transition = dict(transition)
        episode_transition["_success"] = False
        episode.append(episode_transition)

        episode_return += float(reward)
        episode_length += 1
        episode_success = max(episode_success, float(info.get("success", 0.0)))
        done = bool(terminated or truncated)
        if done:
            if episode:
                episode[-1]["_success"] = bool(episode_success > 0.0)
                route_metrics = route_episode_to_buffers(
                    episode,
                    demo_buffer=context.buffers.demo,
                    intervention_buffer=context.buffers.intervention,
                    include_failed_interventions=include_failed_interventions,
                    demo_insert_mode=demo_insert_mode,
                )
            episode_count += 1
            recent_returns.append(episode_return)
            recent_lengths.append(episode_length)
            recent_successes.append(float(episode_success > 0.0))
            recent_returns = recent_returns[-100:]
            recent_lengths = recent_lengths[-100:]
            recent_successes = recent_successes[-100:]
            observation, _ = context.env.reset()
            episode = []
            episode_return = 0.0
            episode_length = 0
            episode_success = 0.0
        else:
            observation = next_observation

        step_update_metrics: dict[str, float] = {}
        if step >= start_training:
            for update_spec in context.update_specs:
                try:
                    step_update_metrics.update(run_update_spec(context, update_spec))
                except ValueError as exc:
                    if "smaller than sequence_length" not in str(exc):
                        raise

        now = time.time()
        force_log = log_interval > 0 and step % log_interval == 0
        payload = {
            "train/step": step,
            "train/online_size": context.buffers.online.size,
            "train/demo_size": context.buffers.demo.size,
            "train/intervention_size": context.buffers.intervention.size,
            "train/episodes": episode_count,
            "train/total_sps": step / max(now - start_time, 1e-6),
            "env/recent_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
            "env/recent_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
            "env/recent_success_rate": float(np.mean(recent_successes)) if recent_successes else 0.0,
            **{f"routing/{key}": float(value) for key, value in route_metrics.items()},
            **step_update_metrics,
        }
        if force_log:
            payload["train/interval_sps"] = (step - last_log_step) / max(now - last_log_time, 1e-6)
        logger.record(payload, step=step, force_flush=force_log)
        if force_log:
            last_log_step = step
            last_log_time = now

        if eval_interval > 0 and step % eval_interval == 0:
            eval_metrics = _evaluate_policy(context, step=step)
            if eval_metrics:
                logger.log_immediate(eval_metrics, step=step, print_stdout=True)

        if save_interval > 0 and step % save_interval == 0:
            _save_train_state(context, step)

    _save_train_state(context, steps)
    _save_buffers(context)
    logger.close()
    print(f"[train] finished run_dir={context.paths.run_dir}", flush=True)
    return context
