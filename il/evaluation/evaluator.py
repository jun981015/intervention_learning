from __future__ import annotations

"""Policy evaluation helpers."""

from pathlib import Path
from typing import Any

import jax
import numpy as np

from il.builders.types import TrainContext
from il.loops.rollout import (
    policy_observation,
    reset_rollout_state,
    residual_policy_observation,
    resolve_residual_scale,
    sample_base_action,
    uses_residual_composition,
)
from il.utils.types import PolicyOutput


class _ContextEvalPolicy:
    """Adapter from the current train context to the generic eval policy API."""

    def __init__(self, context: TrainContext):
        self.context = context

    def reset_episode(self) -> None:
        reset_rollout_state(self.context)

    def sample_action(self, observation, *, rng) -> PolicyOutput:
        context = self.context
        policy_obs = policy_observation(observation, context)
        if uses_residual_composition(context):
            if context.base is None or context.base.policy is None:
                raise ValueError("Residual evaluation requires a base policy.")
            rng, base_rng, action_rng = jax.random.split(rng, 3)
            base_output = sample_base_action(context, observation, rng=base_rng)
            residual_obs = residual_policy_observation(policy_obs, base_output.action)
            output = context.learner.policy.sample_action(residual_obs, rng=action_rng)
            residual_scale = resolve_residual_scale(context)
            action = np.asarray(base_output.action, dtype=np.float32) + residual_scale * np.asarray(
                output.action,
                dtype=np.float32,
            )
            return PolicyOutput(
                action=np.clip(action, -1.0, 1.0),
                info={"base_action": base_output.action, "residual_action": output.action},
            )
        return context.learner.policy.sample_action(policy_obs, rng=rng)


def _reset_env(env, *, seed: int):
    """Reset a Gymnasium-like env while tolerating older reset signatures."""
    try:
        result = env.reset(options={"seed": seed})
    except TypeError:
        try:
            result = env.reset(seed=seed)
        except TypeError:
            result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _reset_policy_episode(policy: Any) -> None:
    reset_episode = getattr(policy, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def _sample_policy_action(policy: Any, observation, *, rng) -> np.ndarray:
    if hasattr(policy, "sample_action"):
        output = policy.sample_action(observation, rng=rng)
    elif callable(policy):
        try:
            output = policy(observation, rng=rng)
        except TypeError:
            output = policy(observation)
    else:
        raise TypeError("Policy must expose sample_action(observation, rng=...) or be callable.")

    action = output.action if hasattr(output, "action") else output
    return np.asarray(action, dtype=np.float32)


def _render_frame(env) -> np.ndarray:
    render = getattr(env, "render", None)
    if not callable(render):
        raise ValueError("Video evaluation requires an env with render().")
    try:
        frame = render(mode="rgb_array")
    except TypeError:
        frame = render()
    if frame is None:
        raise ValueError("env.render() returned None during video evaluation.")
    return np.asarray(frame, dtype=np.uint8)


def _write_video(frames: list[np.ndarray], path: Path, *, fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames were rendered for video output {path}.")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def evaluate_policy(
    policy: Any,
    env: Any,
    *,
    episodes: int,
    seed: int = 0,
    video_dir: str | Path | None = None,
    video_episodes: int = 0,
    video_prefix: str = "eval",
    video_frame_skip: int = 1,
    fps: int = 30,
) -> dict[str, float]:
    """Evaluate a policy in an env, optionally saving rendered video episodes.

    Metric episodes and video episodes are separate. Video episodes are rolled
    out after metric episodes and are not included in return/length/success
    statistics.
    """
    episodes = int(episodes)
    video_episodes = int(video_episodes)
    video_frame_skip = int(video_frame_skip)
    if episodes < 0:
        raise ValueError("episodes must be non-negative.")
    if video_episodes < 0:
        raise ValueError("video_episodes must be non-negative.")
    if video_frame_skip <= 0:
        raise ValueError("video_frame_skip must be positive.")
    if episodes == 0 and video_episodes == 0:
        return {}

    if video_episodes > 0 and video_dir is None:
        raise ValueError("video_dir is required when video_episodes > 0.")
    video_path_root = Path(video_dir) if video_episodes > 0 else None
    if video_path_root is not None and not callable(getattr(env, "render", None)):
        raise ValueError("Video evaluation requires an env with render().")

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    videos_saved = 0
    rng = jax.random.PRNGKey(int(seed))

    total_episodes = episodes + video_episodes
    for episode_idx in range(total_episodes):
        record_video = episode_idx >= episodes
        video_idx = episode_idx - episodes
        frames: list[np.ndarray] = []

        _reset_policy_episode(policy)
        observation, _ = _reset_env(env, seed=int(seed) + episode_idx)
        done = False
        episode_return = 0.0
        episode_length = 0
        episode_success = 0.0

        while not done:
            rng, action_rng = jax.random.split(rng)
            action = _sample_policy_action(policy, observation, rng=action_rng)
            observation, reward, terminated, truncated, info = env.step(np.clip(action, -1.0, 1.0))
            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_length += 1
            episode_success = max(episode_success, float(info.get("success", 0.0)))

            if record_video and (episode_length % video_frame_skip == 0 or done):
                frames.append(_render_frame(env))

        if record_video:
            assert video_path_root is not None
            video_path = video_path_root / f"{video_prefix}_episode_{video_idx:03d}.mp4"
            _write_video(frames, video_path, fps=fps)
            videos_saved += 1
        else:
            returns.append(episode_return)
            lengths.append(episode_length)
            successes.append(episode_success)

    metrics: dict[str, float] = {}
    if returns:
        metrics.update(
            {
                "eval/return": float(np.mean(returns)),
                "eval/length": float(np.mean(lengths)),
                "eval/success_rate": float(np.mean(successes)),
            }
        )
    if video_episodes > 0:
        metrics["eval/video_episodes_saved"] = float(videos_saved)
    return metrics


def evaluate_context_policy(context: TrainContext, *, step: int) -> dict[str, float]:
    """Evaluate the learner policy from a TrainContext using the generic evaluator."""
    eval_env = context.eval_env
    if eval_env is None or context.learner.policy is None:
        return {}

    train_cfg = context.config["train"]
    episodes = int(train_cfg.get("eval_episodes", 0))
    render_video = bool(train_cfg.get("eval_render_video", False))
    video_episodes = int(train_cfg.get("eval_video_episodes", 0)) if render_video else 0
    if episodes <= 0 and video_episodes <= 0:
        return {}

    seed = int(context.config["run"]["seed"]) + int(step)
    video_dir = context.paths.run_dir / "videos" if video_episodes > 0 else None
    saved_rollout_state = context.rollout_state
    context.rollout_state = {}
    try:
        return evaluate_policy(
            _ContextEvalPolicy(context),
            eval_env,
            episodes=episodes,
            seed=seed,
            video_dir=video_dir,
            video_episodes=video_episodes,
            video_prefix=f"eval_step_{int(step)}",
            video_frame_skip=int(train_cfg.get("eval_video_frame_skip", 1)),
        )
    finally:
        context.rollout_state = saved_rollout_state
