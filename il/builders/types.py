from __future__ import annotations

"""Dataclasses shared by training builders."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunPaths:
    """Filesystem locations for one training run."""

    run_dir: Path
    log_dir: Path
    pid_dir: Path


@dataclass(frozen=True)
class EnvSpec:
    """Observation/action structure inferred from a built environment."""

    observation_space: Any
    action_space: Any
    observation_example: Any
    action_example: np.ndarray
    obs_kind: str
    action_dim: int
    obs_dim: int | None = None
    state_key: str | None = None
    pixel_keys: tuple[str, ...] = ()


@dataclass
class ActorBundle:
    """Learner/expert object built from recipe."""

    name: str
    kind: str
    agent: Any | None
    policy: Any | None
    config: dict[str, Any]
    metadata: dict[str, Any]
    checkpoint_path: Path | None
    train: bool


@dataclass
class TrainContext:
    """Objects needed immediately before entering the env-step loop."""

    config: dict[str, Any]
    paths: RunPaths
    env: Any
    eval_env: Any | None
    learner: ActorBundle
    base: ActorBundle | None
    expert: ActorBundle | None
    buffers: Any
    gate: Any | None
    rng: Any
    gate_rng: np.random.Generator
    update_specs: list[dict[str, Any]]
    env_spec: EnvSpec
    obs_dim: int | None
    action_dim: int
    rollout_state: dict[str, Any] = field(default_factory=dict)
