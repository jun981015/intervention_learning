from __future__ import annotations

"""Unified recipe-driven train entrypoint.

`train.py` only orchestrates construction. YAML decides what kind of learner,
expert, replay, gate, update recipe, and routing policy to build. The actual
env-step loop lives in `il.loops.train_loop`.
"""

import argparse
from pathlib import Path

from il.builders.config import configure_runtime_env


configure_runtime_env()

import jax
import numpy as np

from il.builders.actors import build_actor_bundle
from il.builders.components import build_buffers, build_envs, build_gate, infer_env_spec
from il.builders.config import load_recipe, make_run_paths, write_resolved_config
from il.builders.types import TrainContext
from il.loops.train_loop import run_train_loop


def build_context(config: dict) -> TrainContext:
    """Build train context before entering the env-step loop."""
    seed = int(config["run"]["seed"])
    batch_size = int(config["train"]["batch_size"])

    paths = make_run_paths(config)

    env, eval_env = build_envs(config)
    env_spec = infer_env_spec(env)
    obs_dim = env_spec.obs_dim
    action_dim = env_spec.action_dim

    learner = build_actor_bundle(
        name="learner",
        spec=config["learner"],
        env_spec=env_spec,
        batch_size=batch_size,
        seed=seed,
    )

    expert_spec = config.get("expert")
    expert = None
    rollout_execute = config["rollout"].get("execute", "learner")
    should_build_expert = bool(config["rollout"].get("sample_expert", False)) or rollout_execute in ("expert", "gate")
    if expert_spec is not None and should_build_expert:
        expert = build_actor_bundle(
            name="expert",
            spec=expert_spec,
            env_spec=env_spec,
            batch_size=batch_size,
            seed=seed + 1,
        )

    buffers = build_buffers(config, env_spec=env_spec)
    gate = build_gate(config)

    context = TrainContext(
        config=config,
        paths=paths,
        env=env,
        eval_env=eval_env,
        learner=learner,
        expert=expert,
        buffers=buffers,
        gate=gate,
        rng=jax.random.PRNGKey(seed),
        gate_rng=np.random.default_rng(seed),
        update_specs=list(config.get("updates", [])),
        env_spec=env_spec,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    write_resolved_config(config, context)

    print(
        "[train] built context "
        f"run_dir={context.paths.run_dir} "
        f"env={context.config['env']['name']} "
        f"learner={context.learner.kind} "
        f"expert={context.expert.kind if context.expert else 'none'} "
        f"gate={context.config['gate'].get('kind', 'none')} "
        f"obs_kind={context.env_spec.obs_kind} "
        f"obs_dim={context.obs_dim} action_dim={context.action_dim}",
        flush=True,
    )
    return context


def train(config: dict, *, build_only: bool = False) -> TrainContext:
    """Build components and optionally run the configured online train loop."""
    context = build_context(config)
    if build_only:
        print("[train] build-only mode; stopping before rollout.", flush=True)
        return context
    return run_train_loop(context)


def parse_args(argv=None):
    """Parse unified train builder arguments."""
    parser = argparse.ArgumentParser(description="Build training context from a YAML recipe.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML recipe path.")
    parser.add_argument("--build-only", action="store_true", help="Only build components; do not run rollout.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """CLI entrypoint."""
    args = parse_args(argv)
    train(load_recipe(args.config), build_only=args.build_only)


if __name__ == "__main__":
    main()
