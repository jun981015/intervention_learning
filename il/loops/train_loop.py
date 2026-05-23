from __future__ import annotations

"""Online train loop orchestration.

This module owns the env-step loop. Builders construct components, algorithms
own gradient updates, and buffers own storage/sampling.
"""

import time

import numpy as np

from il.buffers.routing import route_episode_to_buffers
from il.buffers.schema import step_record_to_transition
from il.builders.types import TrainContext
from il.evaluation import evaluate_policy
from il.logger import MetricLogger
from il.loops.rollout import choose_rollout_action
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
    payload: dict[str, float] = {
        "gate/expert_execute_rate": float(decision.controller_id == ControllerId.EXPERT),
        "gate/learner_execute_rate": float(decision.controller_id == ControllerId.LEARNER),
        "gate/reason": float(int(decision.reason)),
        "gate/score": float(decision.score),
    }
    for key, value in decision.info.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            payload[f"gate/{key}"] = float(value)
    return payload


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
    route_totals = _empty_route_metrics()

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
            **{f"routing/{key}_total": float(value) for key, value in route_totals.items()},
            **_gate_metric_payload(decision),
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
    _save_buffers(context)
    logger.close()
    print(f"[train] finished run_dir={context.paths.run_dir}", flush=True)
    return context
