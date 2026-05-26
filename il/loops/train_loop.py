from __future__ import annotations

"""Online train loop orchestration.

This module owns the env-step loop. Builders construct components, algorithms
own gradient updates, and buffers own storage/sampling.
"""

import time

import jax
import numpy as np

from il.buffers.routing import route_episode_to_buffers
from il.buffers.schema import step_record_to_transition
from il.builders.types import TrainContext
from il.evaluation import evaluate_policy
from il.logger import MetricLogger
from il.loops.rollout import choose_rollout_action, policy_observation, prepare_next_base_action, reset_rollout_state
from il.loops.updates import run_update_spec
from il.utils.flax_utils import save_agent
from il.utils.types import ControllerId, GateDecision, StepRecord


_ROUTE_METRIC_KEYS = (
    "demo_added",
    "demo_removed",
    "demo_skipped",
    "intervention_added",
    "failed_intervention_seen",
)


def _empty_route_metrics() -> dict[str, int]:
    """Return zero-valued routing event metrics for one env step."""
    return {key: 0 for key in _ROUTE_METRIC_KEYS}


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


def _gate_metric_payload(decision: GateDecision) -> dict[str, float]:
    """Flatten one gate decision into scalar logging metrics."""
    expert_execute = float(decision.controller_id == ControllerId.EXPERT)
    intervention_started = float(decision.info.get("intervention_started", 0.0))
    payload: dict[str, float] = {
        "gate/expert_execute_rate": expert_execute,
        "gate/learner_execute_rate": float(decision.controller_id == ControllerId.LEARNER),
        "gate/expert_execute_steps": expert_execute,
        "gate/intervention_started_count": intervention_started,
        "gate/reason": float(int(decision.reason)),
        "gate/score": float(decision.score),
    }
    for key, value in decision.info.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            payload[f"gate/{key}"] = float(value)
    return payload


def _flatten_numeric_array(value) -> np.ndarray | None:
    """Return a flat fp32 numeric array for health summaries."""
    if isinstance(value, dict):
        parts = [_flatten_numeric_array(item) for item in value.values()]
        parts = [part for part in parts if part is not None and part.size > 0]
        if not parts:
            return None
        return np.concatenate(parts, axis=0)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return None
    if array.ndim > 1:
        return None
    return array.astype(np.float32, copy=False).reshape(-1)


def _array_health(prefix: str, value) -> dict[str, float]:
    """Summarize action/state arrays as fp32 scalar health metrics."""
    array = _flatten_numeric_array(value)
    if array is None or array.size == 0:
        return {}
    finite = np.isfinite(array)
    finite_fraction = float(finite.mean())
    if not bool(finite.any()):
        return {
            f"{prefix}/finite_fraction": finite_fraction,
            f"{prefix}/numel": float(array.size),
        }
    finite_values = array[finite].astype(np.float32, copy=False)
    return {
        f"{prefix}/mean": float(np.mean(finite_values, dtype=np.float32)),
        f"{prefix}/std": float(np.std(finite_values, dtype=np.float32)),
        f"{prefix}/min": float(np.min(finite_values)),
        f"{prefix}/max": float(np.max(finite_values)),
        f"{prefix}/norm": float(np.linalg.norm(finite_values)),
        f"{prefix}/finite_fraction": finite_fraction,
        f"{prefix}/numel": float(array.size),
    }


def _rollout_health_metrics(observation, action: np.ndarray, learner_output, expert_output) -> dict[str, float]:
    """Return low-volume action/state health metrics for one rollout step."""
    metrics: dict[str, float] = {}
    metrics.update(_array_health("state/observation", observation))
    metrics.update(_array_health("action/executed", action))
    metrics.update(_array_health("action/learner", learner_output.action))
    metrics.update(_array_health("action/expert", expert_output.action))

    learner_action = np.asarray(learner_output.action, dtype=np.float32).reshape(-1)
    expert_action = np.asarray(expert_output.action, dtype=np.float32).reshape(-1)
    if learner_action.shape == expert_action.shape and np.isfinite(learner_action).all() and np.isfinite(expert_action).all():
        diff = learner_action - expert_action
        metrics["action/learner_expert_l2"] = float(np.linalg.norm(diff))
        metrics["action/learner_expert_linf"] = float(np.max(np.abs(diff)))
    return metrics


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
    reset_rollout_state(context)
    episode: list[dict] = []
    episode_return = 0.0
    episode_length = 0
    episode_success = 0.0
    episode_count = 0
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    recent_successes: list[float] = []
    route_totals = _empty_route_metrics()
    expert_execute_total = 0
    intervention_started_total = 0

    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    print(f"[train] starting loop steps={steps} run_dir={context.paths.run_dir}", flush=True)

    for step in range(1, steps + 1):
        route_metrics = _empty_route_metrics()
        action, learner_output, expert_output, decision = choose_rollout_action(
            context,
            observation,
            step=step,
        )
        next_observation, reward, terminated, truncated, info = context.env.step(action)
        base_action = None
        residual_action = None
        next_base_action = None
        if context.config["rollout"].get("execute") == "residual":
            base_action = learner_output.info.get("base_action")
            residual_action = learner_output.info.get("residual_action", learner_output.action)
            context.rng, next_base_rng = jax.random.split(context.rng)
            next_base_action = prepare_next_base_action(context, next_observation, rng=next_base_rng).action
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
                base_action=base_action,
                residual_action=residual_action,
                next_base_action=next_base_action,
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
                for key, value in route_metrics.items():
                    route_totals[key] = route_totals.get(key, 0) + int(value)
            episode_count += 1
            recent_returns.append(episode_return)
            recent_lengths.append(episode_length)
            recent_successes.append(float(episode_success > 0.0))
            recent_returns = recent_returns[-100:]
            recent_lengths = recent_lengths[-100:]
            recent_successes = recent_successes[-100:]
            observation, _ = context.env.reset()
            reset_rollout_state(context)
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
                    step_update_metrics.update(run_update_spec(context, update_spec, step=step))
                except ValueError as exc:
                    if "smaller than sequence_length" not in str(exc):
                        raise

        gate_metrics = _gate_metric_payload(decision)
        expert_execute_total += int(gate_metrics.get("gate/expert_execute_steps", 0.0))
        intervention_started_total += int(gate_metrics.get("gate/intervention_started_count", 0.0))

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
            **{f"routing/{key}_total": float(value) for key, value in route_totals.items()},
            **gate_metrics,
            "gate/expert_execute_steps_total": float(expert_execute_total),
            "gate/intervention_started_total": float(intervention_started_total),
            **_rollout_health_metrics(policy_observation(observation, context), action, learner_output, expert_output),
            **_array_health("action/base", learner_output.info.get("base_action")),
            **_array_health("action/residual", learner_output.info.get("residual_action")),
            **_array_health("action/raw_residual", learner_output.info.get("raw_residual_action")),
            **step_update_metrics,
        }
        if force_log:
            payload["train/interval_sps"] = (step - last_log_step) / max(now - last_log_time, 1e-6)
        logger.record(payload, step=step, force_flush=force_log)
        if force_log:
            last_log_step = step
            last_log_time = now

        if eval_interval > 0 and step % eval_interval == 0:
            eval_metrics = evaluate_policy(context, step=step)
            if eval_metrics:
                logger.log_immediate(eval_metrics, step=step, print_stdout=True)

        if save_interval > 0 and step % save_interval == 0:
            _save_train_state(context, step)

    _save_train_state(context, steps)
    if bool(context.recipe.train.get("save_replay", True)):
        _save_buffers(context)
    logger.close()
    print(f"[train] finished run_dir={context.paths.run_dir}", flush=True)
    return context
