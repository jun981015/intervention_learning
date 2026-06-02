from __future__ import annotations

"""Dataset adapters that assign semantic meaning to loaded replay fields."""

from typing import Any

import numpy as np

from il.utils.types import ControllerId, GateReason


STRUCTURAL_REPLAY_KEYS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "masks",
    "next_observations",
)


def _require_keys(dataset: dict[str, Any], keys: tuple[str, ...], *, adapter: str) -> None:
    """Fail fast when an adapter cannot build its declared schema."""
    missing = [key for key in keys if key not in dataset]
    if missing:
        raise KeyError(f"Dataset adapter {adapter!r} requires missing keys: {missing}")


def _dataset_size(dataset: dict[str, Any]) -> int:
    """Return transition count from the canonical action array."""
    return int(np.asarray(dataset["actions"]).shape[0])


def _nan_like_actions(actions: np.ndarray) -> np.ndarray:
    """Return action-shaped NaN labels for unavailable proposals."""
    return np.full_like(actions, np.nan, dtype=np.float32)


def _canonicalize_demo_actions_are_expert(dataset: dict[str, Any]) -> dict[str, Any]:
    """Interpret raw dataset actions as offline expert demonstrations."""
    _require_keys(
        dataset,
        ("observations", "actions", "rewards", "terminals", "masks", "next_observations"),
        adapter="demo_actions_are_expert",
    )
    dataset = dict(dataset)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    n = _dataset_size(dataset)

    dataset["actions"] = actions
    dataset["expert_actions"] = actions.copy()
    dataset["learner_actions"] = _nan_like_actions(actions)
    dataset.setdefault("base_actions", _nan_like_actions(actions))
    dataset.setdefault("residual_actions", _nan_like_actions(actions))
    dataset.setdefault("next_base_actions", _nan_like_actions(actions))
    dataset.setdefault("controller_ids", np.full(n, int(ControllerId.EXPERT), dtype=np.int8))
    dataset.setdefault("episode_ids", np.zeros(n, dtype=np.int64))
    dataset.setdefault("episode_steps", np.arange(n, dtype=np.int32))
    dataset.setdefault("gating_reasons", np.full(n, int(GateReason.NONE), dtype=np.int16))
    dataset.setdefault("gating_scores", np.full(n, np.nan, dtype=np.float32))
    dataset.setdefault("learner_action_log_probs", np.full(n, np.nan, dtype=np.float32))
    dataset.setdefault("expert_action_log_probs", np.full(n, np.nan, dtype=np.float32))
    dataset.setdefault("interventions", np.zeros(n, dtype=np.int8))
    return dataset


def _canonicalize_replay_npz(dataset: dict[str, Any]) -> dict[str, Any]:
    """Validate replay-like data and fill optional metadata conservatively."""
    _require_keys(dataset, STRUCTURAL_REPLAY_KEYS, adapter="replay_npz")
    dataset = dict(dataset)
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    n = _dataset_size(dataset)

    dataset["actions"] = actions
    dataset.setdefault("learner_actions", _nan_like_actions(actions))
    dataset.setdefault("expert_actions", _nan_like_actions(actions))
    dataset.setdefault("base_actions", _nan_like_actions(actions))
    dataset.setdefault("residual_actions", _nan_like_actions(actions))
    dataset.setdefault("next_base_actions", _nan_like_actions(actions))
    dataset.setdefault("controller_ids", np.full(n, int(ControllerId.LEARNER), dtype=np.int8))
    dataset.setdefault("episode_ids", np.zeros(n, dtype=np.int64))
    dataset.setdefault("episode_steps", np.arange(n, dtype=np.int32))
    dataset.setdefault("gating_reasons", np.full(n, int(GateReason.NONE), dtype=np.int16))
    dataset.setdefault("gating_scores", np.full(n, np.nan, dtype=np.float32))
    dataset.setdefault("learner_action_log_probs", np.full(n, np.nan, dtype=np.float32))
    dataset.setdefault("expert_action_log_probs", np.full(n, np.nan, dtype=np.float32))
    dataset.setdefault("interventions", np.zeros(n, dtype=np.int8))
    return dataset


def canonicalize_prefill_dataset(dataset: dict[str, Any], *, adapter: str) -> dict[str, Any]:
    """Convert a loaded dataset into this repo's canonical replay schema."""
    if adapter == "demo_actions_are_expert":
        return _canonicalize_demo_actions_are_expert(dataset)
    if adapter == "replay_npz":
        return _canonicalize_replay_npz(dataset)
    raise ValueError(
        f"Unsupported replay prefill adapter: {adapter!r}. "
        "Expected one of {'demo_actions_are_expert', 'replay_npz'}."
    )
