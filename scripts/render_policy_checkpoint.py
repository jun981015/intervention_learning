from __future__ import annotations

"""Render rollout videos from one configured policy checkpoint."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ.setdefault("EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])

import imageio
import jax
import numpy as np

from il.builders.actors import build_actor_bundle
from il.builders.components import build_envs, infer_env_spec
from il.builders.config import load_recipe
from il.loops.rollout import policy_observation, reset_rollout_state, residual_policy_observation, sample_base_action


def _episode_matches(success: bool, episode_filter: str) -> bool:
    """Return whether a rendered episode should be saved."""
    if episode_filter == "all":
        return True
    if episode_filter == "success":
        return success
    if episode_filter == "failure":
        return not success
    raise ValueError(f"Unsupported episode_filter: {episode_filter!r}")


def _prepare_config(config_path: Path, checkpoint: Path, args) -> dict:
    """Load config and point the learner policy view at `checkpoint`."""
    config = load_recipe(config_path)
    config["run"]["wandb"] = False
    config["env"]["render_offscreen"] = not args.no_video
    if not args.no_video:
        config["env"]["render_camera_name"] = args.camera
        config["env"]["render_hw"] = (args.render_size, args.render_size)
    config["env"]["build_eval_env"] = False
    config["learner"]["pretrained_path"] = str(checkpoint)
    config["learner"]["checkpoint_step"] = None
    config["learner"]["train"] = False
    config["learner"]["policy_view"] = True
    return config


def _render_frame(env) -> np.ndarray:
    """Render through Gym wrappers."""
    try:
        return env.render(mode="rgb_array")
    except TypeError:
        return env.unwrapped.render(mode="rgb_array")


def _rollout_episode(
    policy,
    env,
    env_spec,
    *,
    seed: int,
    rng,
    base=None,
    residual_scale: float = 1.0,
    save_frames: bool = True,
) -> dict:
    """Collect one policy episode with rendered frames."""
    observation, _ = env.reset(options={"seed": seed})
    context = SimpleNamespace(env_spec=env_spec, action_dim=env_spec.action_dim, base=base, rollout_state={})
    reset_rollout_state(context)
    done = False
    frames = []
    rewards = []
    successes = []
    actions = []
    while not done:
        policy_obs = policy_observation(
            observation,
            context,
        )
        if base is None:
            rng, action_rng = jax.random.split(rng)
            output = policy.sample_action(policy_obs, rng=action_rng)
            action = np.asarray(output.action, dtype=np.float32)
        else:
            rng, base_rng, action_rng = jax.random.split(rng, 3)
            base_output = sample_base_action(context, observation, rng=base_rng)
            residual_obs = residual_policy_observation(policy_obs, base_output.action)
            output = policy.sample_action(residual_obs, rng=action_rng)
            action = np.asarray(base_output.action, dtype=np.float32) + residual_scale * np.asarray(output.action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        if save_frames:
            frames.append(_render_frame(env).astype(np.uint8))
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        successes.append(float(info.get("success", 0.0)))
        done = bool(terminated or truncated)
    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "success": bool(np.max(successes) > 0.0) if successes else False,
    }


def _write_video(frames: np.ndarray, path: Path, *, fps: int, video_skip: int) -> None:
    """Write RGB frames to mp4."""
    writer = imageio.get_writer(path, fps=fps)
    try:
        for t in range(0, len(frames), video_skip):
            writer.append_data(frames[t])
    finally:
        writer.close()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config_path = Path(args.config).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _prepare_config(config_path, checkpoint, args)
    env, _ = build_envs(config)
    env_spec = infer_env_spec(env)
    actor = build_actor_bundle(
        name="learner",
        spec=config["learner"],
        env_spec=env_spec,
        batch_size=int(config["train"]["batch_size"]),
        seed=int(config["run"]["seed"]),
    )
    if actor.policy is None:
        raise ValueError("Learner policy view was not built.")
    base = None
    if config["rollout"].get("execute") == "residual":
        base = build_actor_bundle(
            name="base",
            spec=config["base"],
            env_spec=env_spec,
            batch_size=int(config["train"]["batch_size"]),
            seed=int(config["run"]["seed"]) + 2,
        )
        if base.policy is None:
            raise ValueError("Residual rollout requires a base policy view.")
    residual_scale = float(actor.config.get("residual_scale", 1.0))

    rng = jax.random.PRNGKey(args.seed)
    saved = []
    attempts = []
    attempt = 0
    while len(saved) < args.num_episodes and attempt < args.max_attempts:
        episode_seed = args.seed + attempt
        rng, episode_rng = jax.random.split(rng)
        episode = _rollout_episode(
            actor.policy,
            env,
            env_spec,
            seed=episode_seed,
            rng=episode_rng,
            base=base,
            residual_scale=residual_scale,
            save_frames=not args.no_video,
        )
        success = bool(episode["success"])
        attempt_info = {
            "attempt": attempt,
            "seed": episode_seed,
            "success": success,
            "length": int(len(episode["rewards"])),
            "return": float(episode["rewards"].sum()),
        }
        attempts.append(attempt_info)
        print(
            f"[rollout] attempt={attempt} seed={episode_seed} "
            f"success={int(success)} length={attempt_info['length']} return={attempt_info['return']:.3f}",
            flush=True,
        )
        if _episode_matches(success, args.episode_filter):
            idx = len(saved)
            outcome = "success" if success else "failure"
            stem = f"episode_{idx:02d}_{outcome}_seed{episode_seed}"
            video_path = output_dir / f"{stem}.mp4"
            data_path = output_dir / f"{stem}.npz"
            if args.no_video:
                saved.append({"video": None, "data": None, **attempt_info})
            else:
                _write_video(episode["frames"], video_path, fps=args.fps, video_skip=args.video_skip)
                np.savez_compressed(
                    data_path,
                    actions=episode["actions"],
                    rewards=episode["rewards"],
                    successes=episode["successes"],
                    seed=np.asarray(episode_seed, dtype=np.int64),
                    checkpoint=np.asarray(str(checkpoint)),
                    config=np.asarray(str(config_path)),
                )
                saved.append({"video": str(video_path), "data": str(data_path), **attempt_info})
                print(f"[save] {video_path} {data_path}", flush=True)
        attempt += 1

    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "episode_filter": args.episode_filter,
        "num_episodes": len(saved),
        "attempts": attempts,
        "episodes": saved,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if len(saved) < args.num_episodes:
        raise RuntimeError(f"Saved {len(saved)} episodes, requested {args.num_episodes}.")
    print(f"saved {len(saved)} episodes to {output_dir}", flush=True)


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--episode-filter", choices=("all", "success", "failure"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="sideview")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=1)
    parser.add_argument("--no-video", action="store_true", help="Evaluate without rendering or writing videos.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
