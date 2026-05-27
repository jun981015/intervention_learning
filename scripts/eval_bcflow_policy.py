from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ.setdefault("EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])

import jax
import jax.numpy as jnp
import numpy as np

from il.algo.bc.flow import BCFlowAgent
from il.envs import make_robomimic_env
from il.utils.flax_utils import restore_agent_with_file


def load_json(path: Path) -> dict:
    """Load a JSON file from disk."""
    return json.loads(path.read_text())


def load_bcflow_agent(policy_dir: Path, *, seed: int, checkpoint_step: int | None = None) -> tuple[BCFlowAgent, dict, dict]:
    """Load a BCFlow policy checkpoint from this repo's pretrained layout."""
    config = load_json(policy_dir / "config.json")
    metadata = load_json(policy_dir / "metadata.json")
    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    ex_observations = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)
    agent = BCFlowAgent.create(seed, ex_observations, ex_actions, dict(config))
    step = int(checkpoint_step if checkpoint_step is not None else metadata["checkpoint_step"])
    checkpoint = policy_dir / f"params_{step}.pkl"
    agent = restore_agent_with_file(agent, checkpoint)
    metadata = dict(metadata)
    metadata["checkpoint_step"] = step
    return agent, config, metadata


def sample_action_chunk(agent: BCFlowAgent, observation: np.ndarray, rng, *, action_dim: int, horizon_length: int):
    """Sample one BCFlow action chunk and reshape it into primitive env actions."""
    actions, _ = agent.sample_actions_with_log_prob(
        jnp.asarray(observation, dtype=jnp.float32),
        rng=rng,
    )
    actions = np.asarray(actions, dtype=np.float32).reshape(horizon_length, action_dim)
    return np.clip(actions, -1.0, 1.0)


def eval_episode(agent, env, *, seed: int, action_dim: int, horizon_length: int) -> dict:
    """Run one no-render policy evaluation episode."""
    rng = jax.random.PRNGKey(seed)
    observation, _ = env.reset(options={"seed": seed})
    action_queue: list[np.ndarray] = []
    done = False
    episode_return = 0.0
    length = 0
    success = 0.0
    while not done:
        if not action_queue:
            rng, action_rng = jax.random.split(rng)
            chunk = sample_action_chunk(
                agent,
                observation,
                rng=action_rng,
                action_dim=action_dim,
                horizon_length=horizon_length,
            )
            action_queue.extend([chunk[i] for i in range(len(chunk))])
        action = action_queue.pop(0)
        observation, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        episode_return += float(reward)
        length += 1
        success = max(success, float(info.get("success", 0.0)))
        if done:
            action_queue = []
    return {
        "seed": int(seed),
        "return": float(episode_return),
        "length": int(length),
        "success": float(success),
    }


def main() -> None:
    """Evaluate one BCFlow policy and write aggregate and per-episode JSON."""
    args = parse_args()
    policy_dir = Path(args.policy_dir).expanduser()
    output_path = Path(args.output_json).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agent, config, metadata = load_bcflow_agent(policy_dir, seed=args.seed, checkpoint_step=args.checkpoint_step)
    action_dim = int(metadata["action_dim"])
    horizon_length = int(config["horizon_length"] if config["action_chunking"] else 1)
    env = make_robomimic_env(
        args.env_name,
        seed=args.seed,
        render_offscreen=False,
    )

    start = time.time()
    episodes = []
    for episode_idx in range(args.episodes):
        episode_seed = args.seed + episode_idx
        result = eval_episode(
            agent,
            env,
            seed=episode_seed,
            action_dim=action_dim,
            horizon_length=horizon_length,
        )
        episodes.append(result)
        print(
            f"[eval] {episode_idx + 1}/{args.episodes} seed={episode_seed} "
            f"success={result['success']:.0f} return={result['return']:.1f} length={result['length']}",
            flush=True,
        )

    returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float32)
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.float32)
    successes = np.asarray([episode["success"] for episode in episodes], dtype=np.float32)
    summary = {
        "env_name": args.env_name,
        "policy_dir": str(policy_dir),
        "checkpoint_step": int(metadata["checkpoint_step"]),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "success_rate": float(successes.mean()),
        "return_mean": float(returns.mean()),
        "length_mean": float(lengths.mean()),
        "return_std": float(returns.std()),
        "length_std": float(lengths.std()),
        "wall_time_seconds": float(time.time() - start),
        "episode_results": episodes,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"[summary] success_rate={summary['success_rate']:.3f} "
        f"return_mean={summary['return_mean']:.2f} length_mean={summary['length_mean']:.1f} "
        f"json={output_path}",
        flush=True,
    )


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate one pretrained BCFlow policy on Robomimic Square.")
    parser.add_argument("--env-name", default="square-mh-low_dim")
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
