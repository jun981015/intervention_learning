from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
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
import wandb

from il.algo.bc.flow import BCFlowAgent
from il.buffers.replay_buffer import ReplayBuffer
from il.buffers.schema import make_replay_example, step_record_to_transition
from il.envs import make_robomimic_env
from il.policies.rlpd import RLPDPolicy
from il.utils.config import DAggerConfig
from il.utils.flax_utils import restore_agent_with_file, save_agent
from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput, StepRecord


DEFAULT_LEARNER_DIR = Path("exp/pretrained/bcflow_square_actorln_seed0_1m")
DEFAULT_EXPERT_DIR = Path("exp/pretrained/rlpd_square_bc03_seed0_2m")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    log_dir: Path
    pid_dir: Path


def load_pretrained_config(run_dir: Path) -> tuple[dict, dict]:
    """Load `config.json` and `metadata.json` from a local pretrained weight dir."""
    config = json.loads((run_dir / "config.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    return config, metadata


def make_run_dir(args) -> RunPaths:
    """Create a timestamped run directory under ignored experiment folders."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"sd{args.seed:03d}{timestamp}"
    run_dir = Path(args.save_dir) / args.project / args.run_group / args.env_name / run_name
    log_dir = Path("logs")
    pid_dir = Path("pids")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, log_dir=log_dir, pid_dir=pid_dir)


def tree_to_float_dict(tree: dict) -> dict[str, float]:
    """Convert JAX scalar metrics to plain floats for logging."""
    out = {}
    for key, value in tree.items():
        arr = np.asarray(value)
        if arr.size == 1:
            out[key] = float(arr.reshape(()))
    return out


def json_safe(value):
    """Convert argparse/config objects to JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def sample_learner_chunk(agent: BCFlowAgent, observation: np.ndarray, rng, action_dim: int):
    """Sample a full learner action chunk and reshape it into primitive actions."""
    action, log_prob = agent.sample_actions_with_log_prob(
        jnp.asarray(observation, dtype=jnp.float32),
        rng=rng,
    )
    action = np.asarray(action, dtype=np.float32).reshape(-1, action_dim)
    log_prob = np.asarray(log_prob, dtype=np.float32)
    return np.clip(action, -1.0, 1.0), float(log_prob.reshape(-1)[0]) if log_prob.size else float("nan")


def sample_dagger_bc_batch(replay: ReplayBuffer, *, batch_size: int, horizon_length: int) -> dict:
    """Sample BC Flow batches from online DAgger labels.

    `observations` are start states. `expert_actions` is an H-step action-label
    sequence so a chunked diffusion/flow learner can train against expert labels.
    """
    if replay.size < horizon_length:
        raise ValueError(f"Need at least {horizon_length} transitions, got {replay.size}.")

    if horizon_length == 1:
        batch = replay.sample(batch_size)
        return {
            "observations": batch["observations"],
            "expert_actions": batch["expert_actions"],
            "valid": np.ones((batch_size, 1), dtype=np.float32),
        }

    starts = np.random.randint(replay.size - horizon_length + 1, size=batch_size)
    offsets = np.arange(horizon_length)[None, :]
    idxs = starts[:, None] + offsets
    flat = idxs.reshape(-1)

    expert_actions = replay["expert_actions"][flat].reshape(
        batch_size,
        horizon_length,
        *replay["expert_actions"].shape[1:],
    )
    raw_terminals = replay["terminals"][flat].reshape(batch_size, horizon_length)
    valid = np.ones((batch_size, horizon_length), dtype=np.float32)
    for i in range(1, horizon_length):
        valid[:, i] = valid[:, i - 1] * (1.0 - raw_terminals[:, i - 1])

    return {
        "observations": replay["observations"][starts],
        "expert_actions": expert_actions,
        "valid": valid,
    }


def evaluate_learner(agent: BCFlowAgent, env, *, action_dim: int, episodes: int, seed: int) -> dict:
    """Evaluate the current learner with chunk queue execution."""
    rng = jax.random.PRNGKey(seed)
    returns = []
    lengths = []
    successes = []
    for episode in range(episodes):
        observation, _ = env.reset(options={"seed": seed + episode})
        done = False
        action_queue: list[np.ndarray] = []
        episode_return = 0.0
        episode_length = 0
        success = 0.0
        while not done:
            if not action_queue:
                rng, key = jax.random.split(rng)
                chunk, _ = sample_learner_chunk(agent, observation, key, action_dim)
                action_queue.extend([chunk[i] for i in range(len(chunk))])
            action = action_queue.pop(0)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            episode_length += 1
            success = max(success, float(info.get("success", 0)))
            if done:
                action_queue = []
        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(success)
    return {
        "return": float(np.mean(returns)),
        "length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }


def train(args) -> None:
    """Run DAgger with a BC Flow learner and RLPD expert."""
    paths = make_run_dir(args)
    learner_dir = Path(args.learner_dir)
    expert_dir = Path(args.expert_dir)
    learner_config, learner_metadata = load_pretrained_config(learner_dir)
    expert_config, expert_metadata = load_pretrained_config(expert_dir)

    if learner_metadata["obs_dim"] != expert_metadata["obs_dim"]:
        raise ValueError("Learner/expert obs_dim mismatch.")
    if learner_metadata["action_dim"] != expert_metadata["action_dim"]:
        raise ValueError("Learner/expert action_dim mismatch.")

    obs_dim = int(learner_metadata["obs_dim"])
    action_dim = int(learner_metadata["action_dim"])
    horizon_length = int(learner_config["horizon_length"])

    learner_config = dict(learner_config)
    learner_config["batch_size"] = int(args.batch_size)
    learner_config["target_action_key"] = "expert_actions"
    if args.lr is not None:
        learner_config["lr"] = float(args.lr)

    run_config = {
        "args": json_safe(vars(args)),
        "learner_config": learner_config,
        "learner_metadata": learner_metadata,
        "expert_config": expert_config,
        "expert_metadata": expert_metadata,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "horizon_length": horizon_length,
    }
    (paths.run_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=args.project,
            group=args.run_group,
            name=paths.run_dir.name,
            config=run_config,
            tags=[tag for tag in args.wandb_tags.split(",") if tag],
        )

    env = make_robomimic_env(args.env_name, seed=args.seed, render_offscreen=False)
    eval_env = None
    if args.eval_interval > 0 and args.eval_episodes > 0:
        eval_env = make_robomimic_env(args.env_name, seed=args.seed + 10_000, render_offscreen=False)

    ex_observations = jnp.zeros((args.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((args.batch_size, action_dim), dtype=jnp.float32)
    learner = BCFlowAgent.create(args.seed, ex_observations, ex_actions, learner_config)
    learner = restore_agent_with_file(learner, learner_dir / f"params_{learner_metadata['checkpoint_step']}.pkl")

    expert = RLPDPolicy.from_checkpoint(
        expert_dir / f"params_{expert_metadata['checkpoint_step']}.pkl",
        config=expert_config,
        obs_dim=obs_dim,
        action_dim=action_dim,
        seed=int(expert_metadata["seed"]),
    )
    dagger_config = DAggerConfig(store_expert_action=args.store_expert_action)

    replay = ReplayBuffer.create(
        make_replay_example(
            np.zeros(obs_dim, dtype=np.float32),
            np.zeros(action_dim, dtype=np.float32),
        ),
        size=args.buffer_size,
    )

    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)
    observation, _ = env.reset(options={"seed": args.seed})
    action_queue: list[tuple[np.ndarray, float, int, int]] = []

    episode_return = 0.0
    episode_length = 0
    episode_success = 0.0
    episode_count = 0
    recent_successes: list[float] = []
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    update_info = {}

    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0

    for step in range(1, args.steps + 1):
        if not action_queue:
            rng, learner_key = jax.random.split(rng)
            chunk, log_prob = sample_learner_chunk(learner, observation, learner_key, action_dim)
            chunk_length = len(chunk)
            action_queue.extend(
                [(chunk[i], log_prob, i, chunk_length) for i in range(chunk_length)]
            )

        action, learner_log_prob, chunk_index, chunk_length = action_queue.pop(0)
        rng, expert_key = jax.random.split(rng)
        if dagger_config.store_expert_action:
            expert_output = expert.sample_action(observation, rng=expert_key)
        else:
            expert_output = PolicyOutput(
                action=np.full_like(action, np.nan, dtype=np.float32),
                log_prob=float("nan"),
                info={"not_stored": True},
            )
        learner_output = PolicyOutput(
            action=np.asarray(action, dtype=np.float32),
            log_prob=learner_log_prob,
            info={
                "chunk_index": int(chunk_index),
                "chunk_length": int(chunk_length),
            },
        )
        decision = GateDecision(
            controller_id=ControllerId.LEARNER,
            reason=GateReason.NONE,
            score=0.0,
            info={
                "baseline": "dagger",
                "store_expert_action": bool(dagger_config.store_expert_action),
            },
        )

        next_observation, reward, terminated, truncated, info = env.step(action)
        record = StepRecord(
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
        replay.add_transition(step_record_to_transition(record))

        episode_return += float(reward)
        episode_length += 1
        episode_success = max(episode_success, float(info.get("success", 0)))
        done = bool(terminated or truncated)
        if done:
            episode_count += 1
            recent_returns.append(episode_return)
            recent_lengths.append(episode_length)
            recent_successes.append(episode_success)
            recent_returns = recent_returns[-100:]
            recent_lengths = recent_lengths[-100:]
            recent_successes = recent_successes[-100:]
            observation, _ = env.reset()
            action_queue = []
            episode_return = 0.0
            episode_length = 0
            episode_success = 0.0
        else:
            observation = next_observation

        if step >= args.start_training and replay.size >= max(args.batch_size, horizon_length):
            batch = sample_dagger_bc_batch(
                replay,
                batch_size=args.batch_size,
                horizon_length=horizon_length,
            )
            learner, update_info = learner.update(batch)

        if step % args.log_interval == 0:
            now = time.time()
            interval_sps = (step - last_log_step) / max(now - last_log_time, 1e-6)
            total_sps = step / max(now - start_time, 1e-6)
            log_payload = {
                "train/step": step,
                "train/replay_size": replay.size,
                "train/episodes": episode_count,
                "train/interval_sps": interval_sps,
                "train/total_sps": total_sps,
                "env/recent_return": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "env/recent_length": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                "env/recent_success_rate": float(np.mean(recent_successes)) if recent_successes else 0.0,
            }
            if update_info:
                log_payload.update({f"learner/{k}": v for k, v in tree_to_float_dict(update_info).items()})
            print(
                f"[dagger] step={step}/{args.steps} replay={replay.size} "
                f"episodes={episode_count} recent_success={log_payload['env/recent_success_rate']:.3f} "
                f"sps={interval_sps:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb.log(log_payload, step=step)
            last_log_step = step
            last_log_time = now

        if eval_env is not None and step % args.eval_interval == 0:
            eval_info = evaluate_learner(
                learner,
                eval_env,
                action_dim=action_dim,
                episodes=args.eval_episodes,
                seed=args.seed + step,
            )
            print(f"[eval] step={step} {eval_info}", flush=True)
            if wandb_run is not None:
                wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=step)

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_agent(learner, paths.run_dir, step)

    save_agent(learner, paths.run_dir, args.steps)
    replay.save_npz(paths.run_dir / "online_replay_buffer.npz")
    if wandb_run is not None:
        (paths.run_dir / "wandb_url.txt").write_text(wandb_run.url)
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Square DAgger with BC Flow learner and RLPD expert.")
    parser.add_argument("--env-name", default="square-mh-low_dim")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--start-training", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--learner-dir", type=Path, default=DEFAULT_LEARNER_DIR)
    parser.add_argument("--expert-dir", type=Path, default=DEFAULT_EXPERT_DIR)
    parser.add_argument("--save-dir", default="exp")
    parser.add_argument("--project", default="intervention_learning")
    parser.add_argument("--run-group", default="square-dagger-bcflow-rlpd")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--wandb-tags", default="square,dagger,bcflow,rlpd")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--store-expert-action", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
