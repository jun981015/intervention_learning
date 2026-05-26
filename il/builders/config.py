from __future__ import annotations

"""Recipe, runtime, and run-directory helpers for training construction."""

import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from il.builders.types import RunPaths, TrainContext


DEFAULT_RECIPE: dict[str, Any] = {
    "run": {
        "project": "intervention_learning",
        "group": "square_recipe_v1",
        "name": "",
        "save_dir": "exp",
        "seed": 0,
        "wandb": False,
        "tags": [],
    },
    "env": {
        "kind": "robomimic_lowdim",
        "name": "square-mh-low_dim",
        "observation_mode": "lowdim",
        "render_offscreen": False,
        "reward_scale": 1.0,
        "reward_shift": 0.0,
        "build_eval_env": True,
        "eval_seed_offset": 10_000,
    },
    "train": {
        "steps": 300_000,
        "start_training": 1_000,
        "batch_size": 256,
        "log_interval": 5_000,
        "eval_interval": 50_000,
        "eval_episodes": 10,
        "save_interval": 100_000,
    },
    "learner": {
        "kind": "bc_flow",
        "train": True,
        "policy_view": True,
        "pretrained_path": "exp/pretrained/bcflow_square_actorln_seed0_1m",
        "checkpoint_step": None,
        "config": {
            "horizon_length": 1,
            "action_chunking": False,
            "target_action_key": "expert_actions",
        },
    },
    "base": None,
    "expert": {
        "kind": "rlpd",
        "train": False,
        "policy_view": True,
        "pretrained_path": "exp/pretrained/rlpd_square_bc03_seed0_2m",
        "checkpoint_step": None,
        "config": {
            "horizon_length": 1,
            "action_chunking": False,
        },
    },
    "rollout": {
        "sample_learner": True,
        "sample_expert": True,
        "execute": "learner",
    },
    "gate": {
        "kind": "none",
        "expert_probability": 0.0,
    },
    "replay": {
        "frame_stack": 1,
        "online_size": 500_000,
        "demo_size": 500_000,
        "intervention_size": 500_000,
        "prefill": {},
        "demo_insert_mode": "append",
        "include_failed_interventions": False,
    },
    "updates": [
        {
            "name": "learner_bc",
            "target": "learner",
            "source": "online",
            "target_action_key": "expert_actions",
        }
    ],
}


def configure_runtime_env() -> None:
    """Set conservative runtime defaults before importing JAX/MuJoCo-heavy code."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ.setdefault("EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ["CUDA_VISIBLE_DEVICES"])


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_recipe(path: str | Path | None) -> dict[str, Any]:
    """Load an optional YAML recipe and merge it over `DEFAULT_RECIPE`.

    Public configs that contain `experiment:` use the new schema documented in
    `docs/CONFIG_SCHEMA_DECISIONS.md`; they are converted to the current
    internal recipe shape until the builders are migrated directly.
    """
    if path is None:
        return copy.deepcopy(DEFAULT_RECIPE)
    recipe_path = Path(path).expanduser()
    with recipe_path.open("r") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Recipe must be a YAML mapping, got {type(loaded).__name__}.")
    if "experiment" in loaded:
        recipe = new_schema_to_legacy_recipe(loaded)
    else:
        recipe = deep_update(DEFAULT_RECIPE, loaded)
    recipe["_recipe_path"] = str(recipe_path)
    return recipe


def _map_env_kind(kind: str) -> str:
    """Map public config env kind names to internal env builder keys."""
    if kind == "robomimic":
        return "robomimic_lowdim"
    return kind


def _actor_config_from_new_schema(actor: dict[str, Any]) -> dict[str, Any]:
    """Convert actor network/optimization/update fields to agent config overrides."""
    network = actor.get("network") or {}
    optimization = actor.get("optimization") or {}
    update = actor.get("update") or {}
    config: dict[str, Any] = {}
    config.update(network)
    config.update(optimization)
    config.update(update)

    # Public names are clearer in YAML; internal agent configs retain qc-style names.
    if "critic_hidden_dims" in config and "value_hidden_dims" not in config:
        config["value_hidden_dims"] = config.pop("critic_hidden_dims")
    if "critic_layer_norm" in config and "layer_norm" not in config:
        config["layer_norm"] = config.pop("critic_layer_norm")

    # Activation selection is documented but not wired through network modules yet.
    for reserved in ("activation", "actor_activation", "critic_activation"):
        config.pop(reserved, None)
    return config


def _new_actor_to_legacy(actor: dict[str, Any]) -> dict[str, Any]:
    """Convert one new-schema actor block to the current builder actor spec."""
    return {
        "kind": actor["kind"],
        "train": bool(actor.get("trainable", False)),
        "policy_view": bool(actor.get("policy_view", True)),
        "pretrained_path": actor.get("pretrained_path"),
        "checkpoint_step": actor.get("checkpoint_step"),
        "config": _actor_config_from_new_schema(actor),
    }


def _first_sampling_spec(sampling: dict[str, Any], *, learner_kind: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return the primary named sampling spec used by the current v0 train loop."""
    if learner_kind in {"rlpd", "residual_rlpd", "residual_td3", "acrlpd", "sac", "td3"} and "rl" in sampling:
        return "rl", sampling["rl"]
    if "bc" in sampling:
        return "bc", sampling["bc"]
    if "rl" in sampling:
        return "rl", sampling["rl"]
    if len(sampling) == 1:
        key = next(iter(sampling))
        return key, sampling[key]
    raise ValueError("replay.sampling must contain at least one named sampling spec.")


def _source_to_legacy(source: Any) -> tuple[str, dict[str, float] | None]:
    """Convert sampling source to legacy source/sampling_fractions fields."""
    if isinstance(source, str):
        return source, None
    if isinstance(source, dict):
        if len(source) == 1:
            name, value = next(iter(source.items()))
            if float(value) == 1.0:
                return str(name), None
        return "mixed", {str(key): float(value) for key, value in source.items()}
    raise TypeError(f"Unsupported replay sampling source: {source!r}")


def new_schema_to_legacy_recipe(config: dict[str, Any]) -> dict[str, Any]:
    """Convert the public config schema to the current internal recipe schema."""
    recipe = copy.deepcopy(DEFAULT_RECIPE)
    experiment = config["experiment"]
    env = config["env"]
    actors = config["actors"]
    training = config["training"]
    replay = config["replay"]
    intervention = config["intervention"]
    evaluation = config.get("evaluation") or {}
    logging = config.get("logging") or {}
    checkpointing = config.get("checkpointing") or {}

    wandb_cfg = logging.get("wandb") or {}
    recipe["run"] = {
        "project": wandb_cfg.get("project", "intervention_learning"),
        "group": wandb_cfg.get("group", experiment["name"]),
        "name": wandb_cfg.get("name", experiment.get("name", "")),
        "save_dir": experiment.get("output_dir", "exp/runs"),
        "seed": int(experiment.get("seed", 0)),
        "wandb": bool(wandb_cfg.get("enabled", False)),
        "jsonl": bool(logging.get("jsonl", True)),
        "csv": bool(logging.get("csv", True)),
        "tags": list(wandb_cfg.get("tags", experiment.get("tags", []))),
    }

    recipe["env"] = {
        **{key: value for key, value in env.items() if key != "kind"},
        "kind": _map_env_kind(env["kind"]),
        "build_eval_env": bool(env.get("build_eval_env", True)),
        "eval_seed_offset": int(env.get("eval_seed_offset", 10_000)),
    }

    learner_kind = str(actors["learner"].get("kind", ""))
    all_sampling = replay.get("sampling") or {}
    sampling_name, sampling = _first_sampling_spec(all_sampling, learner_kind=learner_kind)
    batch_size = int(
        sampling.get(
            "batch_size",
            actors["learner"].get("optimization", {}).get("batch_size", DEFAULT_RECIPE["train"]["batch_size"]),
        )
    )

    recipe["train"] = {
        "steps": int(training["total_steps"]),
        "start_training": int(training.get("start_training", 0)),
        "batch_size": batch_size,
        "log_interval": int(logging.get("stdout_interval", DEFAULT_RECIPE["train"]["log_interval"])),
        "eval_interval": int(evaluation.get("interval", 0)),
        "eval_episodes": int(evaluation.get("episodes", 0)),
        "save_interval": int(checkpointing.get("interval", 0)),
        "save_replay": bool(checkpointing.get("save_replay", True)),
    }

    recipe["learner"] = _new_actor_to_legacy(actors["learner"])
    base = actors.get("base")
    recipe["base"] = _new_actor_to_legacy(base) if base is not None else None
    expert = actors.get("expert")
    recipe["expert"] = _new_actor_to_legacy(expert) if expert is not None else None

    intervention_enabled = bool(intervention.get("enabled", False))
    expert_query = intervention.get("expert_query", "always" if intervention_enabled else "never")
    recipe["rollout"] = {
        "sample_learner": True,
        "sample_expert": expert_query == "always",
        "execute": "learner",
        "expert_query": expert_query,
        "action_mode": training.get("action_mode", "first_action"),
        "action_composition": training.get("action_composition", "direct"),
    }
    if training.get("action_composition") == "residual":
        recipe["rollout"]["execute"] = "residual"
    for key in ("residual_warmup_steps", "warmup_noise_scale", "random_action_noise_scale", "use_base_policy_for_warmup"):
        if key in training:
            recipe["rollout"][key] = copy.deepcopy(training[key])
    if intervention_enabled:
        recipe["rollout"]["execute"] = "gate"
        recipe["rollout"]["sample_expert"] = expert_query == "always"

    gate = intervention.get("gate") or {}
    gate_kind = gate.get("kind", "always_off")
    recipe["gate"] = {
        "kind": "none" if gate_kind in ("always_off", "none", None) else gate_kind,
    }
    if recipe["gate"]["kind"] == "random":
        recipe["gate"]["expert_probability"] = float(gate.get("expert_probability", gate.get("probability", 0.0)))
    for key in ("threshold", "intervention_prob", "intervention_horizon", "q_agg"):
        if key in gate:
            recipe["gate"][key] = copy.deepcopy(gate[key])

    buffers = replay.get("buffers") or {}
    routing = replay.get("routing") or {}
    recipe["replay"] = {
        "frame_stack": int(replay.get("frame_stack", 1)),
        "online_size": int(buffers.get("online_size", DEFAULT_RECIPE["replay"]["online_size"])),
        "demo_size": int(buffers.get("demo_size", DEFAULT_RECIPE["replay"]["demo_size"])),
        "intervention_size": int(buffers.get("intervention_size", DEFAULT_RECIPE["replay"]["intervention_size"])),
        "prefill": replay.get("prefill") or {},
        "demo_insert_mode": routing.get("demo_insert_mode", "none"),
        "include_failed_interventions": bool(routing.get("include_failed_interventions", False)),
    }

    source, sampling_fractions = _source_to_legacy(sampling.get("source", "online"))
    update_spec = {
        "name": f"learner_{sampling_name}",
        "target": "learner",
        "source": source,
        "batch_size": batch_size,
        "sequence_length": int(sampling.get("sequence_length", recipe["learner"]["config"].get("td_n_step", 1))),
    }
    for key in ("utd_ratio", "utd", "critic_warmup_steps", "update_actor"):
        if key in sampling:
            update_spec[key] = copy.deepcopy(sampling[key])
    target_action_key = recipe["learner"]["config"].get("target_action_key")
    if target_action_key is not None:
        update_spec["target_action_key"] = target_action_key
    if sampling_fractions is not None:
        update_spec["sampling_fractions"] = sampling_fractions
    if sampling_name != "bc" and "bc" in all_sampling:
        bc_sampling = all_sampling["bc"]
        bc_source, bc_sampling_fractions = _source_to_legacy(bc_sampling.get("source", "demo"))
        bc_spec = {
            "source": bc_source,
            "batch_size": int(bc_sampling.get("batch_size", batch_size)),
            "sequence_length": int(bc_sampling.get("sequence_length", 1)),
        }
        if bc_sampling_fractions is not None:
            bc_spec["sampling_fractions"] = bc_sampling_fractions
        update_spec["aux_batches"] = {"bc": bc_spec}
    recipe["updates"] = [update_spec]

    recipe["_public_schema"] = copy.deepcopy(config)
    return recipe


def make_run_paths(config: dict[str, Any]) -> RunPaths:
    """Create run/log/pid directories for this recipe."""
    run_cfg = config["run"]
    env_cfg = config["env"]
    seed = int(run_cfg["seed"])
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = run_cfg.get("name") or f"sd{seed:03d}{timestamp}"
    run_dir = (
        Path(run_cfg["save_dir"])
        / run_cfg["project"]
        / run_cfg["group"]
        / env_cfg["name"]
        / run_name
    )
    log_dir = Path("logs")
    pid_dir = Path("pids")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, log_dir=log_dir, pid_dir=pid_dir)


def json_safe(value: Any) -> Any:
    """Convert nested runtime objects to JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_resolved_config(config: dict[str, Any], context: TrainContext) -> None:
    """Save the resolved recipe and key runtime paths to the run directory."""
    resolved = copy.deepcopy(config)
    resolved["_resolved"] = {
        "obs_kind": context.env_spec.obs_kind,
        "obs_dim": context.obs_dim,
        "action_dim": context.action_dim,
        "state_key": context.env_spec.state_key,
        "pixel_keys": list(context.env_spec.pixel_keys),
        "learner_checkpoint": str(context.learner.checkpoint_path)
        if context.learner.checkpoint_path
        else None,
        "base_checkpoint": (
            str(context.base.checkpoint_path)
            if context.base is not None and context.base.checkpoint_path is not None
            else None
        ),
        "expert_checkpoint": (
            str(context.expert.checkpoint_path)
            if context.expert is not None and context.expert.checkpoint_path is not None
            else None
        ),
    }
    (context.paths.run_dir / "config.json").write_text(
        json.dumps(json_safe(resolved), indent=2, sort_keys=True)
    )
    context.config = resolved
