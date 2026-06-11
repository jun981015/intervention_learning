from __future__ import annotations

"""Measure action-uncertainty gate statistics for one policy checkpoint."""

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np

from il.builders.actors import build_actor_bundle
from il.builders.config import load_recipe
from il.builders.types import EnvSpec
from il.gating.action_uncertainty import ActionUncertaintyGate
from il.gating.base import GateContext
from il.loops.rollout import _sample_gate_policy_source
from il.utils.types import PolicyOutput


def _infer_step(checkpoint: Path) -> int:
    """Infer the training step from a params_<step>.pkl filename."""
    match = re.search(r"params_(\d+)\.pkl$", checkpoint.name)
    if match is None:
        raise ValueError(f"Cannot infer step from checkpoint name: {checkpoint.name}")
    return int(match.group(1))


def _load_observations(path: Path, *, key: str, num_states: int, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a fixed random observation batch from a replay npz."""
    data = np.load(path)
    if key not in data.files:
        raise KeyError(f"{path} does not contain key={key!r}; available={data.files}")
    observations = np.asarray(data[key], dtype=np.float32)
    size = int(np.asarray(data["size"])) if "size" in data.files else int(observations.shape[0])
    size = min(size, int(observations.shape[0]))
    if size <= 0:
        raise ValueError(f"{path} contains no observations.")
    count = min(int(num_states), size)
    rng = np.random.default_rng(seed)
    indices = rng.choice(size, size=count, replace=False)
    return observations[indices], indices.astype(np.int64), size


def _make_env_spec(*, obs_dim: int, action_dim: int) -> EnvSpec:
    """Create the low-dimensional EnvSpec needed by actor builders."""
    return EnvSpec(
        observation_space=None,
        action_space=None,
        observation_example=np.zeros(obs_dim, dtype=np.float32),
        action_example=np.zeros(action_dim, dtype=np.float32),
        obs_kind="lowdim",
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


def _prepare_recipe(config_path: Path, checkpoint: Path, *, exploration_noise: float) -> dict[str, Any]:
    """Load a training recipe and point the learner policy view at a checkpoint."""
    recipe = load_recipe(config_path)
    recipe["run"]["wandb"] = False
    recipe["learner"]["pretrained_path"] = str(checkpoint)
    recipe["learner"]["checkpoint_step"] = None
    recipe["learner"]["train"] = False
    recipe["learner"]["policy_view"] = True
    recipe["learner"].setdefault("config", {})["exploration_noise"] = float(exploration_noise)
    recipe["rollout"]["execute"] = "gate"
    recipe["rollout"]["action_composition"] = "residual"
    return recipe


def _build_context(recipe: dict[str, Any], *, obs_dim: int, action_dim: int):
    """Build learner/base policy views without constructing an environment."""
    env_spec = _make_env_spec(obs_dim=obs_dim, action_dim=action_dim)
    batch_size = int(recipe["train"].get("batch_size", 256))
    seed = int(recipe["run"].get("seed", 0))
    learner = build_actor_bundle(
        name="learner",
        spec=recipe["learner"],
        env_spec=env_spec,
        batch_size=batch_size,
        seed=seed,
    )
    base = build_actor_bundle(
        name="base",
        spec=recipe["base"],
        env_spec=env_spec,
        batch_size=batch_size,
        seed=seed + 2,
    )
    return SimpleNamespace(
        config=recipe,
        learner=learner,
        base=base,
        expert=None,
        action_dim=action_dim,
        rollout_state={},
    )


def _measure_state(context, observation: np.ndarray, *, step: int, num_samples: int, rng_key) -> dict[str, Any]:
    """Run ActionUncertaintyGate once for one observation."""
    dummy = PolicyOutput(action=np.zeros(context.action_dim, dtype=np.float32))
    gate = ActionUncertaintyGate(
        threshold=float("inf"),
        source="learner",
        estimator="sample_variance",
        num_samples=num_samples,
        score="rms_std",
        intervention_prob=0.0,
        intervention_horizon=1,
    )
    policy_obs = np.asarray(observation, dtype=np.float32)

    def sample_policy(source: str) -> PolicyOutput:
        nonlocal rng_key
        rng_key, sample_rng = jax.random.split(rng_key)
        return _sample_gate_policy_source(
            context,
            source,
            policy_obs,
            step=step,
            rng=sample_rng,
            learner_output=dummy,
        )

    gate_context = GateContext(
        sample_policy=sample_policy,
        policy_observation=policy_obs,
        action_dim=context.action_dim,
    )
    decision = gate.decide(
        step=step,
        observation=policy_obs,
        learner=dummy,
        expert=dummy,
        rng=np.random.default_rng(step),
        context=gate_context,
    )
    return {
        "score": float(decision.score),
        "var": decision.info["var"],
        "action_variance_mean": float(decision.info["action_variance_mean"]),
        "action_variance_max": float(decision.info["action_variance_max"]),
        "action_std_mean": float(decision.info["action_std_mean"]),
        "action_std_max": float(decision.info["action_std_max"]),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-state uncertainty rows."""
    scores = np.asarray([row["score"] for row in rows], dtype=np.float32)
    var_means = np.asarray([row["var"]["mean"] for row in rows], dtype=np.float32)
    var_maxes = np.asarray([row["var"]["max"] for row in rows], dtype=np.float32)
    per_dim = np.asarray([row["var"]["per_dim"] for row in rows], dtype=np.float32)
    return {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p95": float(np.percentile(scores, 95)),
        "score_p99": float(np.percentile(scores, 99)),
        "score_max": float(np.max(scores)),
        "var_mean_mean": float(np.mean(var_means)),
        "var_mean_p95": float(np.percentile(var_means, 95)),
        "var_max_mean": float(np.mean(var_maxes)),
        "var_max_p95": float(np.percentile(var_maxes, 95)),
        "per_dim_var_mean": np.mean(per_dim, axis=0).astype(np.float32).tolist(),
        "per_dim_var_p95": np.percentile(per_dim, 95, axis=0).astype(np.float32).tolist(),
    }


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    config_path = Path(args.config).expanduser()
    replay_buffer = Path(args.replay_buffer).expanduser() if args.replay_buffer else checkpoint.parent / "demo_replay_buffer.npz"
    step = int(args.step) if args.step is not None else _infer_step(checkpoint)

    observations, indices, replay_size = _load_observations(
        replay_buffer,
        key=args.observation_key,
        num_states=args.num_states,
        seed=args.seed,
    )
    action_dim = int(np.load(replay_buffer)["actions"].shape[-1])
    obs_dim = int(observations.shape[-1])
    recipe = _prepare_recipe(config_path, checkpoint, exploration_noise=args.exploration_noise)
    context = _build_context(recipe, obs_dim=obs_dim, action_dim=action_dim)

    rng_key = jax.random.PRNGKey(args.seed)
    rows = []
    for local_index, observation in enumerate(observations):
        rng_key, state_rng = jax.random.split(rng_key)
        row = _measure_state(
            context,
            observation,
            step=step,
            num_samples=args.num_samples,
            rng_key=state_rng,
        )
        row["replay_index"] = int(indices[local_index])
        rows.append(row)

    result = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_step": step,
        "replay_buffer": str(replay_buffer),
        "replay_size": replay_size,
        "observation_key": args.observation_key,
        "num_states": int(len(rows)),
        "num_samples": int(args.num_samples),
        "source": "learner",
        "metric": "final_composed_action_variance",
        "exploration_noise_override": float(args.exploration_noise),
        "summary": _summarize(rows),
        "states": rows,
    }

    output = Path(args.output).expanduser() if args.output else _default_output(checkpoint, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({**result, "states": f"{len(rows)} rows omitted", "output": str(output)}, indent=2, sort_keys=True))


def _default_output(checkpoint: Path, args) -> Path:
    """Return a stable diagnostic output path."""
    run_name = checkpoint.parent.name
    return Path("diagnostics/action_uncertainty") / (
        f"{run_name}_{checkpoint.stem}_n{args.num_states}_s{args.num_samples}_noise{args.exploration_noise:g}.json"
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--replay-buffer", default=None)
    parser.add_argument("--observation-key", default="observations")
    parser.add_argument("--num-states", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--exploration-noise", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
