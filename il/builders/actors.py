from __future__ import annotations

"""Learner/expert agent builders."""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from il.algo.bc.flow import BCFlowAgent, get_config as get_bc_flow_config
from il.algo.bc.mlp import BCMLPAgent, get_config as get_bc_mlp_config
from il.algo.rl.residual_rlpd import ResidualRLPDAgent, get_config as get_residual_rlpd_config
from il.algo.rl.residual_td3 import ResidualTD3Agent, get_config as get_residual_td3_config
from il.algo.rl.rlpd import ACRLPDAgent, get_config as get_rlpd_config
from il.builders.config import deep_update
from il.builders.types import ActorBundle, EnvSpec
from il.policies.agent_view import AgentPolicyView
from il.utils.flax_utils import restore_agent_with_file


def _to_plain_dict(config: Any) -> dict[str, Any]:
    """Convert ml_collections ConfigDict-like objects to plain dictionaries."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    return dict(config)


def default_agent_config(kind: str) -> dict[str, Any]:
    """Return a mutable default config for one supported agent kind."""
    if kind == "bc_flow":
        return _to_plain_dict(get_bc_flow_config())
    if kind == "bc_mlp":
        return _to_plain_dict(get_bc_mlp_config())
    if kind == "rlpd":
        cfg = _to_plain_dict(get_rlpd_config())
        cfg["target_entropy"] = None
        return cfg
    if kind == "residual_rlpd":
        cfg = _to_plain_dict(get_residual_rlpd_config())
        cfg["target_entropy"] = None
        cfg.setdefault("residual_scale", 0.1)
        cfg.setdefault("residual_action_l2", 0.0)
        return cfg
    if kind == "residual_td3":
        cfg = _to_plain_dict(get_residual_td3_config())
        cfg.setdefault("residual_scale", 0.2)
        cfg.setdefault("residual_action_l2", 0.0)
        return cfg
    raise ValueError(f"Unsupported actor kind: {kind!r}")


def load_pretrained_state(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    """Load optional pretrained config/metadata and resolve a checkpoint path."""
    if "artifact_dir" in spec:
        raise ValueError("Use `pretrained_path` instead of the old `artifact_dir` key.")

    pretrained_path = spec.get("pretrained_path")
    if not pretrained_path:
        return {}, {}, None

    path = Path(pretrained_path).expanduser()
    run_dir = path.parent if path.is_file() else path

    config_path = run_dir / "config.json"
    metadata_path = run_dir / "metadata.json"
    pretrained_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    if path.is_file():
        checkpoint_path = path
    else:
        step = spec.get("checkpoint_step")
        if step is None:
            step = metadata.get("checkpoint_step")
        if step is None:
            raise ValueError(f"Set checkpoint_step or metadata.checkpoint_step for pretrained_path={path}")
        checkpoint_path = path / f"params_{int(step)}.pkl"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")
    return dict(pretrained_config), dict(metadata), checkpoint_path


def resolve_agent_config(
    *,
    kind: str,
    spec: dict[str, Any],
    pretrained_config: dict[str, Any],
    env_spec: EnvSpec,
    batch_size: int,
) -> dict[str, Any]:
    """Resolve defaults + pretrained config + YAML overrides into one config."""
    if env_spec.obs_dim is None:
        raise NotImplementedError(
            f"{kind} currently requires a low-dim state input. "
            f"Got obs_kind={env_spec.obs_kind!r}; add an image encoder before training this agent."
        )

    config = default_agent_config(kind)
    config = deep_update(config, pretrained_config)
    config = deep_update(config, spec.get("config", {}))
    config["batch_size"] = int(config.get("batch_size") or batch_size)
    config["horizon_length"] = int(config.get("horizon_length") or 1)
    config["action_dim"] = int(env_spec.action_dim)
    if kind == "bc_flow":
        config["ob_dims"] = tuple(config.get("ob_dims") or (env_spec.obs_dim,))
    if kind in {"residual_rlpd", "residual_td3"}:
        config["residual_policy"] = True
        config["base_obs_dim"] = int(env_spec.obs_dim)
        if bool(config.get("action_chunking", False)):
            raise NotImplementedError(f"{kind} v0 supports primitive actions only; set action_chunking=False.")
    return config


def create_agent(
    kind: str,
    *,
    seed: int,
    env_spec: EnvSpec,
    batch_size: int,
    config: dict[str, Any],
):
    """Create an un-restored trainable agent."""
    if env_spec.obs_dim is None:
        raise NotImplementedError("Image observations need an encoder-backed agent implementation.")
    obs_dim = int(env_spec.obs_dim)
    ex_observations = jnp.zeros((batch_size, obs_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((batch_size, env_spec.action_dim), dtype=jnp.float32)
    if kind == "bc_flow":
        return BCFlowAgent.create(seed, ex_observations, ex_actions, config)
    if kind == "bc_mlp":
        return BCMLPAgent.create(seed, ex_observations, ex_actions, config)
    if kind == "rlpd":
        return ACRLPDAgent.create(seed, ex_observations, ex_actions, config)
    if kind == "residual_rlpd":
        return ResidualRLPDAgent.create(seed, ex_observations, ex_actions, config)
    if kind == "residual_td3":
        return ResidualTD3Agent.create(seed, ex_observations, ex_actions, config)
    raise ValueError(f"Unsupported trainable actor kind: {kind!r}")


def maybe_restore_agent(agent, checkpoint_path: Path | None):
    """Restore an agent if a pretrained checkpoint was specified."""
    if checkpoint_path is None:
        return agent
    return restore_agent_with_file(agent, checkpoint_path)


def build_actor_bundle(
    *,
    name: str,
    spec: dict[str, Any],
    env_spec: EnvSpec,
    batch_size: int,
    seed: int,
) -> ActorBundle:
    """Build trainable agent and optional policy view for learner/expert."""
    kind = spec["kind"]
    if name in {"expert", "base"} and not spec.get("pretrained_path"):
        label = name.capitalize()
        raise ValueError(f"{label} requires `pretrained_path`; refusing to create a random-init {name} actor.")

    pretrained_config, metadata, checkpoint_path = load_pretrained_state(spec)
    if metadata:
        expected_obs_dim = env_spec.obs_dim
        if kind in {"residual_rlpd", "residual_td3"} and expected_obs_dim is not None:
            expected_obs_dim = int(expected_obs_dim) + int(env_spec.action_dim)
        if expected_obs_dim is not None and int(metadata.get("obs_dim", expected_obs_dim)) != expected_obs_dim:
            raise ValueError(f"{name} pretrained obs_dim does not match expected obs_dim.")
        if int(metadata.get("action_dim", env_spec.action_dim)) != env_spec.action_dim:
            raise ValueError(f"{name} pretrained action_dim does not match env action_dim.")

    config = resolve_agent_config(
        kind=kind,
        spec=spec,
        pretrained_config=pretrained_config,
        env_spec=env_spec,
        batch_size=batch_size,
    )
    train = bool(spec.get("train", False))
    policy_view = bool(spec.get("policy_view", True))
    agent = None
    if train or policy_view:
        agent = create_agent(
            kind,
            seed=seed,
            env_spec=env_spec,
            batch_size=batch_size,
            config=config,
        )
        agent = maybe_restore_agent(agent, checkpoint_path)

    policy = None
    if policy_view:
        if agent is None:
            raise ValueError(f"{name} requested policy_view=True but no agent exists.")
        policy = AgentPolicyView(
            agent=agent,
            kind=kind,
            checkpoint_path=checkpoint_path,
            obs_dim=int(env_spec.obs_dim) + (env_spec.action_dim if kind in {"residual_rlpd", "residual_td3"} else 0),
            action_dim=env_spec.action_dim,
            horizon_length=int(config["horizon_length"]),
            action_chunking=bool(config.get("action_chunking", False)),
        )

    return ActorBundle(
        name=name,
        kind=kind,
        agent=agent,
        policy=policy,
        config=config,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
        train=train,
    )
