from __future__ import annotations

"""Small builders for env, replay buffers, and gates."""

from typing import Any

import numpy as np
from gymnasium import spaces

from il.buffers import ReplayBuffer, ReplayBufferCollection, load_npz_dataset, make_replay_example
from il.builders.types import EnvSpec
from il.gating import RandomGate


def _flat_box_dim(space: spaces.Space, *, name: str) -> int:
    """Return flattened dimension for Box spaces used by low-dim policies."""
    if not isinstance(space, spaces.Box):
        raise TypeError(f"{name} must be a gymnasium.spaces.Box, got {type(space).__name__}.")
    if space.shape is None:
        raise ValueError(f"{name} has no shape.")
    return int(np.prod(space.shape))


def _zero_from_space(space: spaces.Space):
    """Create one zero observation/action example matching a Gymnasium space."""
    if isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    if isinstance(space, spaces.Dict):
        return {key: _zero_from_space(subspace) for key, subspace in space.spaces.items()}
    raise TypeError(f"Unsupported space type: {type(space).__name__}")


def _classify_observation_space(space: spaces.Space) -> tuple[str, int | None, str | None, tuple[str, ...]]:
    """Classify observation structure without flattening image observations."""
    if isinstance(space, spaces.Box):
        if space.shape is None:
            raise ValueError("observation_space has no shape.")
        if len(space.shape) == 1:
            return "lowdim", _flat_box_dim(space, name="observation_space"), None, ()
        if len(space.shape) == 3:
            return "pixels", None, None, ("pixels",)
        raise TypeError(f"Unsupported Box observation shape: {space.shape}")

    if isinstance(space, spaces.Dict):
        state_keys = []
        pixel_keys = []
        for key, subspace in space.spaces.items():
            if not isinstance(subspace, spaces.Box) or subspace.shape is None:
                raise TypeError(f"Unsupported observation subspace for key={key!r}: {subspace}")
            if len(subspace.shape) == 1:
                state_keys.append(key)
            elif len(subspace.shape) == 3:
                pixel_keys.append(key)
            else:
                raise TypeError(f"Unsupported observation shape for key={key!r}: {subspace.shape}")

        if len(state_keys) > 1:
            raise ValueError(f"Expected at most one low-dim state key, got {state_keys}")
        state_key = state_keys[0] if state_keys else None
        obs_dim = _flat_box_dim(space.spaces[state_key], name=f"observation_space[{state_key}]") if state_key else None
        if pixel_keys and state_key:
            return "pixels_state", obs_dim, state_key, tuple(pixel_keys)
        if pixel_keys:
            return "pixels", None, None, tuple(pixel_keys)
        if state_key:
            return "dict_lowdim", obs_dim, state_key, ()
    raise TypeError(f"Unsupported observation_space: {space}")


def infer_env_spec(env) -> EnvSpec:
    """Infer observation/action structure from a built environment."""
    observation_space = getattr(env, "single_observation_space", None) or getattr(
        env, "observation_space", None
    )
    action_space = getattr(env, "single_action_space", None) or getattr(env, "action_space", None)
    if observation_space is None or action_space is None:
        raise AttributeError("Env must expose observation_space/action_space.")
    action_dim = _flat_box_dim(action_space, name="action_space")
    obs_kind, obs_dim, state_key, pixel_keys = _classify_observation_space(observation_space)
    return EnvSpec(
        observation_space=observation_space,
        action_space=action_space,
        observation_example=_zero_from_space(observation_space),
        action_example=_zero_from_space(action_space),
        obs_kind=obs_kind,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_key=state_key,
        pixel_keys=pixel_keys,
    )


def build_envs(config: dict[str, Any]):
    """Build training and optional eval environments from config."""
    from il.envs import make_env

    env_cfg = config["env"]
    seed = int(config["run"]["seed"])
    env = make_env(env_cfg, seed=seed)
    eval_env = None
    if bool(env_cfg.get("build_eval_env", True)) and int(config["train"].get("eval_interval", 0)) > 0:
        eval_env = make_env(
            env_cfg,
            seed=seed + int(env_cfg.get("eval_seed_offset", 10_000)),
        )
    return env, eval_env


def build_buffers(config: dict[str, Any], *, env_spec: EnvSpec) -> ReplayBufferCollection:
    """Build online/demo/intervention replay buffers."""
    replay_cfg = config["replay"]
    example = make_replay_example(
        env_spec.observation_example,
        env_spec.action_example,
    )
    frame_stack = int(replay_cfg.get("frame_stack", 1))
    prefill_cfg = replay_cfg.get("prefill") or {}

    def build_one(name: str, size_key: str) -> ReplayBuffer:
        size = int(replay_cfg[size_key])
        spec = prefill_cfg.get(name)
        if spec is None:
            return ReplayBuffer.create(example, size, frame_stack=frame_stack)
        if isinstance(spec, str):
            spec = {"path": spec}
        fmt = spec.get("format", "npz")
        if fmt != "npz":
            raise ValueError(f"Unsupported replay prefill format for {name}: {fmt!r}")
        dataset = load_npz_dataset(
            spec["path"],
            max_transitions=spec.get("max_transitions"),
        )
        return ReplayBuffer.create_from_initial_dataset(dataset, size, frame_stack=frame_stack)

    return ReplayBufferCollection(
        online=build_one("online", "online_size"),
        demo=build_one("demo", "demo_size"),
        intervention=build_one("intervention", "intervention_size"),
    )


def build_gate(config: dict[str, Any]):
    """Build optional gate function."""
    gate_cfg = config["gate"]
    kind = gate_cfg.get("kind", "none")
    if kind in ("none", None):
        return None
    if kind == "random":
        return RandomGate(expert_probability=float(gate_cfg["expert_probability"]))
    raise ValueError(f"Unsupported gate kind: {kind!r}")
