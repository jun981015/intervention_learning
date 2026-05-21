from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from il.algo.bc.flow import BCFlowAgent, get_config
from il.envs.robomimic_lowdim import LOW_DIM_KEYS
from il.utils.flax_utils import restore_agent_with_file, save_agent


DEFAULT_DATASET = Path("~/.robomimic/square/mh/low_dim_v15.hdf5")


def json_safe(value):
    """Convert config/argparse values to JSON-compatible Python values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [json_safe(val) for val in value]
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def tree_to_float_dict(tree: dict) -> dict[str, float]:
    """Convert scalar JAX metrics to plain floats for logging."""
    out = {}
    for key, value in tree.items():
        arr = np.asarray(value)
        if arr.size == 1:
            out[key] = float(arr.reshape(()))
    return out


def demo_sort_key(demo_name: str) -> int:
    """Sort `demo_10` after `demo_9` rather than lexicographically."""
    try:
        return int(demo_name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def concat_lowdim_obs(obs_group, obs_keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate Robomimic low-dimensional observation keys."""
    return np.concatenate([obs_group[key][()] for key in obs_keys], axis=-1).astype(np.float32)


def load_topk_sequences(
    dataset_path: Path,
    *,
    top_k: int,
    horizon_length: int,
    obs_keys: tuple[str, ...] = LOW_DIM_KEYS["low_dim"],
) -> tuple[dict[str, np.ndarray], dict]:
    """Load shortest `top_k` Robomimic demos as state/action chunk training examples."""
    dataset_path = dataset_path.expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    observations = []
    action_chunks = []
    selected = []
    all_lengths = []
    with h5py.File(dataset_path, "r") as file:
        demo_names = sorted(file["data"].keys(), key=demo_sort_key)
        for demo_name in demo_names:
            length = int(file["data"][demo_name]["actions"].shape[0])
            all_lengths.append((demo_name, length))

        sorted_by_length = sorted(all_lengths, key=lambda item: (item[1], demo_sort_key(item[0])))
        selected = sorted_by_length[:top_k]
        for demo_name, length in selected:
            demo = file["data"][demo_name]
            if length < horizon_length:
                continue
            demo_obs = concat_lowdim_obs(demo["obs"], obs_keys)
            demo_actions = demo["actions"][()].astype(np.float32)
            for start in range(length - horizon_length + 1):
                observations.append(demo_obs[start])
                action_chunks.append(demo_actions[start : start + horizon_length])

    if not observations:
        raise ValueError(f"No valid {horizon_length}-step chunks found for top_k={top_k}.")

    metadata = {
        "dataset_path": str(dataset_path),
        "top_k": int(top_k),
        "selection_rule": "shortest_successful_demos_by_action_length",
        "selected_demos": [
            {"name": demo_name, "length": int(length)}
            for demo_name, length in selected
        ],
        "num_demos_total": len(all_lengths),
        "num_chunks": len(observations),
        "min_selected_length": int(min(length for _, length in selected)),
        "max_selected_length": int(max(length for _, length in selected)),
        "obs_keys": list(obs_keys),
    }
    data = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(action_chunks, dtype=np.float32),
        "valid": np.ones((len(observations), horizon_length), dtype=np.float32),
    }
    return data, metadata


def sample_batch(data: dict[str, np.ndarray], *, batch_size: int, rng: np.random.Generator) -> dict:
    """Sample an i.i.d. chunk batch from preloaded top-K demo arrays."""
    idxs = rng.integers(0, data["observations"].shape[0], size=batch_size)
    return {
        "observations": data["observations"][idxs],
        "actions": data["actions"][idxs],
        "valid": data["valid"][idxs],
    }


def make_run_dir(args) -> Path:
    """Return the output directory for one top-K BCFlow pretraining run."""
    run_name = args.run_name or f"bcflow_square_top{args.top_k}_actorln_seed{args.seed}_{args.steps // 1000}k"
    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_checkpoint_step(run_dir: Path) -> int:
    """Return the largest `params_<step>.pkl` step in `run_dir`."""
    steps = []
    for path in run_dir.glob("params_*.pkl"):
        match = re.fullmatch(r"params_(\d+)\.pkl", path.name)
        if match is not None:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No params_<step>.pkl checkpoints found in {run_dir}")
    return max(steps)


def build_config(args, *, obs_dim: int, action_dim: int) -> dict:
    """Build BCFlow config matching the existing pretrained policy layout."""
    config = get_config()
    config.ob_dims = (obs_dim,)
    config.action_dim = action_dim
    config.batch_size = args.batch_size
    config.horizon_length = args.horizon_length
    config.action_chunking = True
    config.actor_layer_norm = args.actor_layer_norm
    config.actor_hidden_dims = tuple(args.actor_hidden_dims)
    config.flow_steps = args.flow_steps
    config.lr = args.lr
    config.grad_clip_norm = args.grad_clip_norm
    config.weight_decay = args.weight_decay
    config.actor_type = "flow"
    config.target_action_key = "actions"
    config.use_fourier_features = args.use_fourier_features
    config.fourier_feature_dim = args.fourier_feature_dim
    return dict(config)


def train(args) -> None:
    """Train a BCFlow policy on the shortest top-K Robomimic Square demos."""
    run_dir = make_run_dir(args)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    data, dataset_metadata = load_topk_sequences(
        args.dataset,
        top_k=args.top_k,
        horizon_length=args.horizon_length,
    )
    obs_dim = int(data["observations"].shape[-1])
    action_dim = int(data["actions"].shape[-1])
    config = build_config(args, obs_dim=obs_dim, action_dim=action_dim)

    ex_observations = jnp.zeros((args.batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((args.batch_size, action_dim), dtype=jnp.float32)
    agent = BCFlowAgent.create(args.seed, ex_observations, ex_actions, config)

    start_step = 0
    resume_metadata = None
    if args.resume_dir is not None:
        resume_dir = args.resume_dir.expanduser()
        start_step = int(args.resume_step or latest_checkpoint_step(resume_dir))
        if start_step >= args.steps:
            raise ValueError(f"Resume step {start_step} is already >= target steps {args.steps}.")
        resume_path = resume_dir / f"params_{start_step}.pkl"
        agent = restore_agent_with_file(agent, resume_path)
        resume_metadata = {
            "resume_dir": str(resume_dir),
            "resume_step": start_step,
            "resume_checkpoint": str(resume_path),
        }

    run_metadata = {
        "seed": int(args.seed),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "full_action_dim": action_dim * int(args.horizon_length),
        "checkpoint_step": int(args.steps),
        "dataset": dataset_metadata,
        "resume": resume_metadata,
    }
    (run_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2, sort_keys=True))
    (run_dir / "metadata.json").write_text(json.dumps(json_safe(run_metadata), indent=2, sort_keys=True))

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=args.project,
            group=args.run_group,
            name=args.wandb_name or run_dir.name,
            config={
                "args": json_safe(vars(args)),
                "bc_flow_config": json_safe(config),
                "metadata": json_safe(run_metadata),
            },
            tags=[tag for tag in args.wandb_tags.split(",") if tag],
        )

    np_rng = np.random.default_rng(args.seed)
    start_time = time.time()
    last_log_time = start_time
    last_log_step = start_step
    last_info = {}

    for step in range(start_step + 1, args.steps + 1):
        batch = sample_batch(data, batch_size=args.batch_size, rng=np_rng)
        agent, last_info = agent.update(batch)

        if step % args.log_interval == 0 or step == start_step + 1:
            now = time.time()
            interval_sps = (step - last_log_step) / max(now - last_log_time, 1e-6)
            total_sps = step / max(now - start_time, 1e-6)
            metrics = {
                "train/step": step,
                "train/top_k": args.top_k,
                "train/num_chunks": dataset_metadata["num_chunks"],
                "train/interval_sps": interval_sps,
                "train/total_sps": total_sps,
            }
            metrics.update({f"learner/{key}": value for key, value in tree_to_float_dict(last_info).items()})
            print(
                f"[bcflow-topk] top_k={args.top_k} step={step}/{args.steps} "
                f"resume={start_step} "
                f"loss={metrics.get('learner/actor/bc_flow_loss', float('nan')):.6f} "
                f"sps={interval_sps:.2f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb.log(metrics, step=step)
            last_log_step = step
            last_log_time = now

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_agent(agent, run_dir, step)
            run_metadata["checkpoint_step"] = int(step)
            (run_dir / "metadata.json").write_text(json.dumps(json_safe(run_metadata), indent=2, sort_keys=True))

    save_agent(agent, run_dir, args.steps)
    run_metadata["checkpoint_step"] = int(args.steps)
    run_metadata["wall_time_seconds"] = float(time.time() - start_time)
    (run_dir / "metadata.json").write_text(json.dumps(json_safe(run_metadata), indent=2, sort_keys=True))
    if wandb_run is not None:
        (run_dir / "wandb_url.txt").write_text(wandb_run.url)
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain BCFlow on shortest top-K Robomimic Square demos.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--resume-dir", type=Path, default=None)
    parser.add_argument("--resume-step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon-length", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--actor-hidden-dims", type=int, nargs="+", default=[512, 512, 512, 512])
    parser.add_argument("--actor-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-fourier-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fourier-feature-dim", type=int, default=64)
    parser.add_argument("--save-dir", type=Path, default=Path("exp/pretrained"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--project", default="intervention_learning")
    parser.add_argument("--run-group", default="square-bcflow-topk-pretrain")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-tags", default="square,bcflow,pretrain,topk")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
