from __future__ import annotations

"""Collect failed policy rollouts and render videos with action/state traces."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ.setdefault("EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])

import imageio
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from il.algo.bc.flow import BCFlowAgent
from il.envs import make_robomimic_env
from il.utils.flax_utils import restore_agent_with_file


DEFAULT_POLICY_DIR = Path("exp/pretrained/bcflow_square_actorln_seed0_1m")
STATE_GROUP_LABELS = (
    "eef_pos",
    "eef_quat",
    "gripper",
    "object",
)


def load_json(path: Path) -> dict:
    """Load one JSON file."""
    return json.loads(path.read_text())


def load_bcflow_agent(policy_dir: Path, *, seed: int) -> tuple[BCFlowAgent, dict, dict]:
    """Load this repo's pretrained BCFlow diffusion/flow policy."""
    config = load_json(policy_dir / "config.json")
    metadata = load_json(policy_dir / "metadata.json")
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    ex_observations = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)
    agent = BCFlowAgent.create(seed, ex_observations, ex_actions, dict(config))
    checkpoint = policy_dir / f"params_{int(metadata['checkpoint_step'])}.pkl"
    agent = restore_agent_with_file(agent, checkpoint)
    return agent, config, metadata


def sample_action_chunk(agent: BCFlowAgent, observation: np.ndarray, rng, *, action_dim: int, horizon_length: int):
    """Sample a full BCFlow action chunk and reshape to primitive actions."""
    action, log_prob = agent.sample_actions_with_log_prob(
        jnp.asarray(observation, dtype=jnp.float32),
        rng=rng,
    )
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    chunk = action.reshape(horizon_length, action_dim)
    log_prob = np.asarray(log_prob, dtype=np.float32).reshape(-1)
    return np.clip(chunk, -1.0, 1.0), float(log_prob[0]) if log_prob.size else float("nan")


def collect_episode(agent, env, *, seed: int, action_dim: int, horizon_length: int):
    """Roll out one episode and return raw arrays plus rendered frames."""
    rng = jax.random.PRNGKey(seed)
    observation, _ = env.reset(options={"seed": seed})
    done = False
    action_queue: list[tuple[np.ndarray, float, int]] = []
    frames = []
    states = []
    actions = []
    rewards = []
    successes = []
    log_probs = []
    chunk_indices = []

    while not done:
        if not action_queue:
            rng, action_rng = jax.random.split(rng)
            chunk, log_prob = sample_action_chunk(
                agent,
                observation,
                action_dim=action_dim,
                horizon_length=horizon_length,
                rng=action_rng,
            )
            action_queue.extend((chunk[i], log_prob, i) for i in range(len(chunk)))

        action, log_prob, chunk_index = action_queue.pop(0)
        frames.append(render_rgb_array(env).astype(np.uint8))
        states.append(np.asarray(observation, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        log_probs.append(log_prob)
        chunk_indices.append(chunk_index)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        successes.append(float(info.get("success", 0.0)))
        done = bool(terminated or truncated)

    return {
        "frames": np.asarray(frames, dtype=np.uint8),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "successes": np.asarray(successes, dtype=np.float32),
        "log_probs": np.asarray(log_probs, dtype=np.float32),
        "chunk_indices": np.asarray(chunk_indices, dtype=np.int32),
        "success": bool(np.max(successes) > 0.0) if successes else False,
    }


def render_rgb_array(env) -> np.ndarray:
    """Render through gym wrappers that may not forward render kwargs."""
    try:
        return env.render(mode="rgb_array")
    except TypeError:
        if hasattr(env, "unwrapped"):
            return env.unwrapped.render(mode="rgb_array")
        return env.env.render(mode="rgb_array")


def normalize_state_for_plot(states: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    """Return plottable state traces and axis label."""
    if mode == "raw":
        return states, "state"
    if mode != "zscore":
        raise ValueError(f"Unsupported state plot mode: {mode}")
    mean = states.mean(axis=0, keepdims=True)
    std = states.std(axis=0, keepdims=True)
    return (states - mean) / np.maximum(std, 1e-6), "state z-score"


def plot_panel(actions: np.ndarray, states: np.ndarray, *, t: int, panel_hw: tuple[int, int], state_mode: str):
    """Render action/state traces into an RGB image panel."""
    panel_h, panel_w = panel_hw
    dpi = 100
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(panel_w / dpi, panel_h / dpi),
        dpi=dpi,
        sharex=True,
        constrained_layout=True,
    )
    xs = np.arange(len(actions))
    action_colors = plt.cm.tab10(np.linspace(0, 1, actions.shape[1]))
    for dim in range(actions.shape[1]):
        axes[0].plot(xs, actions[:, dim], color=action_colors[dim], linewidth=1.3, label=f"a{dim}")
    axes[0].axvline(t, color="black", linewidth=1.2)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_ylabel("action")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=4, fontsize=7, loc="upper right")

    states_for_plot, state_ylabel = normalize_state_for_plot(states, state_mode)
    for dim in range(states_for_plot.shape[1]):
        axes[1].plot(xs, states_for_plot[:, dim], color="tab:blue", linewidth=0.55, alpha=0.35)
    axes[1].plot(xs, states_for_plot.mean(axis=1), color="black", linewidth=1.4, label="dim mean")
    axes[1].axvline(t, color="black", linewidth=1.2)
    ylim = np.nanpercentile(np.abs(states_for_plot), 98)
    ylim = max(float(ylim), 1.0)
    axes[1].set_ylim(-ylim, ylim)
    axes[1].set_ylabel(state_ylabel)
    axes[1].set_xlabel("timestep")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="upper right")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    panel = np.asarray(rgba[..., :3], dtype=np.uint8)
    plt.close(fig)
    return panel


def write_composite_video(episode: dict, path: Path, *, fps: int, video_skip: int, state_mode: str) -> None:
    """Write a side-by-side render + action/state trace video."""
    frames = episode["frames"]
    states = episode["states"]
    actions = episode["actions"]
    render_h, render_w = frames[0].shape[:2]
    panel_w = max(render_w * 2, 512)
    writer = imageio.get_writer(path, fps=fps)
    try:
        for t in range(0, len(frames), video_skip):
            panel = plot_panel(
                actions,
                states,
                t=t,
                panel_hw=(render_h, panel_w),
                state_mode=state_mode,
            )
            writer.append_data(np.concatenate([frames[t], panel], axis=1))
    finally:
        writer.close()


def save_episode_data(episode: dict, path: Path, *, seed: int, attempt: int, camera: str) -> None:
    """Save raw state/action traces for later analysis."""
    np.savez_compressed(
        path,
        states=episode["states"],
        actions=episode["actions"],
        rewards=episode["rewards"],
        successes=episode["successes"],
        log_probs=episode["log_probs"],
        chunk_indices=episode["chunk_indices"],
        seed=np.asarray(seed, dtype=np.int64),
        attempt=np.asarray(attempt, dtype=np.int64),
        camera=np.asarray(camera),
    )


def main() -> None:
    """Collect failed episodes and write videos/data files."""
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = Path(args.policy_dir).expanduser()
    agent, config, metadata = load_bcflow_agent(policy_dir, seed=args.seed)
    action_dim = int(metadata["action_dim"])
    horizon_length = int(config["horizon_length"] if config["action_chunking"] else 1)

    env = make_robomimic_env(
        args.env_name,
        seed=args.seed,
        render_offscreen=True,
        render_camera_name=args.camera,
        render_hw=(args.render_size, args.render_size),
    )

    failures = []
    summary = {
        "policy_dir": str(policy_dir),
        "env_name": args.env_name,
        "camera": args.camera,
        "seed": args.seed,
        "target_failures": args.num_failures,
        "attempts": [],
    }
    attempt = 0
    while len(failures) < args.num_failures and attempt < args.max_attempts:
        episode_seed = args.seed + attempt
        episode = collect_episode(
            agent,
            env,
            seed=episode_seed,
            action_dim=action_dim,
            horizon_length=horizon_length,
        )
        success = bool(episode["success"])
        summary["attempts"].append(
            {
                "attempt": attempt,
                "seed": episode_seed,
                "success": success,
                "length": int(len(episode["actions"])),
                "return": float(episode["rewards"].sum()),
            }
        )
        print(
            f"[rollout] attempt={attempt} seed={episode_seed} "
            f"success={int(success)} length={len(episode['actions'])}",
            flush=True,
        )
        if not success:
            failure_idx = len(failures)
            stem = f"failure_{failure_idx:02d}_seed{episode_seed}"
            video_path = output_dir / f"{stem}.mp4"
            data_path = output_dir / f"{stem}.npz"
            write_composite_video(
                episode,
                video_path,
                fps=args.fps,
                video_skip=args.video_skip,
                state_mode=args.state_plot_mode,
            )
            save_episode_data(
                episode,
                data_path,
                seed=episode_seed,
                attempt=attempt,
                camera=args.camera,
            )
            failures.append({"video": str(video_path), "data": str(data_path), "seed": episode_seed})
            print(f"[save] {video_path} {data_path}", flush=True)
        attempt += 1

    summary["failures"] = failures
    summary["num_failures"] = len(failures)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if len(failures) < args.num_failures:
        raise RuntimeError(
            f"Collected only {len(failures)} failures after {args.max_attempts} attempts. "
            f"See {output_dir / 'summary.json'}."
        )
    print(f"saved {len(failures)} failure episodes to {output_dir}", flush=True)


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="square-mh-low_dim")
    parser.add_argument("--policy-dir", default=str(DEFAULT_POLICY_DIR))
    parser.add_argument("--output-dir", default="videos/failure_rollouts/bcflow_square_actorln_seed0_1m")
    parser.add_argument("--num-failures", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="sideview")
    parser.add_argument("--render-size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-skip", type=int, default=2)
    parser.add_argument("--state-plot-mode", choices=("zscore", "raw"), default="zscore")
    return parser.parse_args()


if __name__ == "__main__":
    main()
