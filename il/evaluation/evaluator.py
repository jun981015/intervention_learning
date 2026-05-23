from __future__ import annotations

"""Policy evaluation helpers."""

import jax
import numpy as np

from il.builders.types import TrainContext
from il.loops.rollout import policy_observation


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
    for episode_idx in range(episodes):
        observation, _ = eval_env.reset(options={"seed": seed + episode_idx})
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_success = 0.0
        while not done:
            rng, action_rng = jax.random.split(rng)
            policy_obs = policy_observation(observation, context)
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
    return {
        "eval/return": float(np.mean(returns)),
        "eval/length": float(np.mean(lengths)),
        "eval/success_rate": float(np.mean(successes)),
    }
