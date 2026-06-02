from __future__ import annotations

"""Robomimic dataset loaders for replay prefill adapters."""

from pathlib import Path

import h5py
import numpy as np

from il.envs.robomimic_lowdim import LOW_DIM_KEYS


def demo_sort_key(demo_name: str) -> int:
    """Sort demo names numerically when they follow `demo_123`."""
    try:
        return int(demo_name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def concat_lowdim_obs(obs_group, obs_keys: tuple[str, ...]) -> np.ndarray:
    """Concatenate robomimic low-dimensional observation keys."""
    return np.concatenate([obs_group[key][()] for key in obs_keys], axis=-1).astype(np.float32)


def load_robomimic_lowdim_replay_dataset(
    path: str | Path,
    *,
    obs_keys: tuple[str, ...] = LOW_DIM_KEYS["low_dim"],
    max_demos: int | None = None,
    max_transitions: int | None = None,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
) -> dict[str, np.ndarray]:
    """Load robomimic low-dim demos without assigning controller semantics.

    Episode boundaries are the HDF5 demo boundaries rather than every positive
    `dones` entry, because robomimic demos may keep several final
    success-labelled frames.
    """
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Robomimic dataset not found: {dataset_path}")

    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    masks = []
    episode_ids = []
    episode_steps = []

    total = 0
    with h5py.File(dataset_path, "r") as file:
        demo_names = sorted(file["data"].keys(), key=demo_sort_key)
        if max_demos is not None:
            demo_names = demo_names[: int(max_demos)]
        for episode_id, demo_name in enumerate(demo_names):
            demo = file["data"][demo_name]
            demo_obs = concat_lowdim_obs(demo["obs"], obs_keys)
            if "next_obs" in demo:
                demo_next_obs = concat_lowdim_obs(demo["next_obs"], obs_keys)
            else:
                demo_next_obs = np.concatenate([demo_obs[1:], demo_obs[-1:]], axis=0)
            demo_actions = demo["actions"][()].astype(np.float32)
            length = int(demo_actions.shape[0])
            raw_rewards = demo["rewards"][()].astype(np.float32) if "rewards" in demo else np.zeros(length, dtype=np.float32)
            demo_rewards = raw_rewards * float(reward_scale) + float(reward_shift)

            if max_transitions is not None:
                remaining = int(max_transitions) - total
                if remaining <= 0:
                    break
                length = min(length, remaining)

            observations.append(demo_obs[:length])
            next_observations.append(demo_next_obs[:length])
            actions.append(demo_actions[:length])
            rewards.append(demo_rewards[:length])
            terminal = np.zeros(length, dtype=np.float32)
            terminal[-1] = 1.0
            terminals.append(terminal)
            masks.append(1.0 - terminal)
            episode_ids.append(np.full(length, episode_id, dtype=np.int64))
            episode_steps.append(np.arange(length, dtype=np.int32))
            total += length
            if max_transitions is not None and total >= int(max_transitions):
                break

    if total == 0:
        raise ValueError(f"No transitions loaded from {dataset_path}")

    observations_arr = np.concatenate(observations, axis=0).astype(np.float32)
    next_observations_arr = np.concatenate(next_observations, axis=0).astype(np.float32)
    actions_arr = np.concatenate(actions, axis=0).astype(np.float32)
    rewards_arr = np.concatenate(rewards, axis=0).astype(np.float32)
    terminals_arr = np.concatenate(terminals, axis=0).astype(np.float32)
    masks_arr = np.concatenate(masks, axis=0).astype(np.float32)
    episode_ids_arr = np.concatenate(episode_ids, axis=0).astype(np.int64)
    episode_steps_arr = np.concatenate(episode_steps, axis=0).astype(np.int32)
    return {
        "observations": observations_arr,
        "actions": actions_arr,
        "rewards": rewards_arr,
        "terminals": terminals_arr,
        "masks": masks_arr,
        "next_observations": next_observations_arr,
        "episode_ids": episode_ids_arr,
        "episode_steps": episode_steps_arr,
    }
