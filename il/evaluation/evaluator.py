from __future__ import annotations

"""Policy evaluation helpers."""

import jax
import numpy as np

from il.builders.types import TrainContext
from il.loops.rollout import policy_observation, reset_rollout_state, residual_policy_observation, sample_base_action


def evaluate_policy(context: TrainContext, *, step: int) -> dict[str, float]:
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
    rng = jax.random.PRNGKey(seed)

    saved_rollout_state = context.rollout_state
    context.rollout_state = {}
    try:
        for episode_idx in range(episodes):
            observation, _ = eval_env.reset(options={"seed": seed + episode_idx})
            reset_rollout_state(context)
            done = False
            episode_return = 0.0
            episode_length = 0
            episode_success = 0.0
            while not done:
                policy_obs = policy_observation(observation, context)
                if context.config["rollout"].get("execute") == "residual":
                    if context.base is None or context.base.policy is None:
                        raise ValueError("Residual evaluation requires a base policy.")
                    rng, base_rng, action_rng = jax.random.split(rng, 3)
                    base_output = sample_base_action(context, observation, rng=base_rng)
                    residual_obs = residual_policy_observation(policy_obs, base_output.action)
                    output = context.learner.policy.sample_action(residual_obs, rng=action_rng)
                    residual_scale = float(context.learner.config.get("residual_scale", 1.0))
                    action = np.asarray(base_output.action, dtype=np.float32) + residual_scale * np.asarray(output.action, dtype=np.float32)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    rng, action_rng = jax.random.split(rng)
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
    finally:
        context.rollout_state = saved_rollout_state

    return {
        "eval/return": float(np.mean(returns)),
        "eval/length": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(successes)),
    }
