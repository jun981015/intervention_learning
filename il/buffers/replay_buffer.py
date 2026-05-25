from __future__ import annotations

"""Replay storage and QC-style n-step sequence sampling.

This module intentionally keeps the buffer numpy-based.  It is used for online,
demo, and intervention streams, so it should not depend on a specific JAX agent
implementation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


def _is_image_value(value) -> bool:
    """Return whether a value looks like an image frame."""
    if isinstance(value, dict):
        return False
    value = np.asarray(value)
    return value.ndim == 3 and value.shape[-1] in (1, 3, 4) and np.issubdtype(value.dtype, np.integer)


def _tree_empty(value) -> bool:
    """Return whether a nested dict tree contains no leaves."""
    if isinstance(value, dict):
        return all(_tree_empty(item) for item in value.values())
    return False


def _split_image_tree(value):
    """Split image leaves from a nested observation tree."""
    if isinstance(value, dict):
        non_images = {}
        images = {}
        for key, item in value.items():
            item_non_images, item_images = _split_image_tree(item)
            if not _tree_empty(item_non_images):
                non_images[key] = item_non_images
            if not _tree_empty(item_images):
                images[key] = item_images
        return non_images, images
    if _is_image_value(value):
        return {}, np.asarray(value)
    return np.asarray(value), {}


def _merge_tree(base, extra):
    """Merge nested dict leaves from `extra` into `base`."""
    if _tree_empty(extra):
        return base
    if _tree_empty(base):
        base = {}
    if not isinstance(base, dict) or not isinstance(extra, dict):
        raise TypeError("Can only merge image observations into dict observation trees.")
    merged = dict(base)
    for key, value in extra.items():
        if key in merged:
            merged[key] = _merge_tree(merged[key], value)
        else:
            merged[key] = value
    return merged


def _tree_len(value) -> int:
    """Return leading dimension length for an array tree."""
    if isinstance(value, dict):
        return max(_tree_len(item) for item in value.values())
    return len(value)


def get_size(data: dict) -> int:
    """Infer dataset length from the longest value array, including nested obs trees."""
    return max(_tree_len(value) for value in data.values())


def _tree_zeros(value, size: int):
    """Allocate a replay array tree from one example value."""
    if isinstance(value, dict):
        return {key: _tree_zeros(item, size) for key, item in value.items()}
    value = np.asarray(value)
    return np.zeros((size, *value.shape), dtype=value.dtype)


def _tree_assign(target, index: int, value) -> None:
    """Assign one transition value into a replay tree."""
    if isinstance(target, dict):
        if set(target) != set(value):
            raise KeyError(
                "Nested replay schema mismatch. "
                f"missing={set(target) - set(value)}, extra={set(value) - set(target)}"
            )
        for key in target:
            _tree_assign(target[key], index, value[key])
        return
    target[index] = value


def _tree_index(value, idxs: np.ndarray):
    """Index a replay tree by batch indices."""
    if isinstance(value, dict):
        return {key: _tree_index(item, idxs) for key, item in value.items()}
    return value[idxs]


def _tree_item(value, index: int):
    """Return one copied item from an array tree."""
    if isinstance(value, dict):
        return {key: _tree_item(item, index) for key, item in value.items()}
    return np.array(value[index], copy=True)


def _tree_zero_in_place(value) -> None:
    """Reset an allocated replay array tree without changing its schema."""
    if isinstance(value, dict):
        for item in value.values():
            _tree_zero_in_place(item)
        return
    value[...] = 0


def _tree_mask_invalid(value, valid: np.ndarray):
    """Zero out batch entries where `valid` is false."""
    if isinstance(value, dict):
        return {key: _tree_mask_invalid(item, valid) for key, item in value.items()}
    mask_shape = (len(valid),) + (1,) * (value.ndim - 1)
    return np.where(valid.reshape(mask_shape), value, np.zeros_like(value))


def _tree_sequence(value, flat_idxs: np.ndarray, batch_size: int, sequence_length: int):
    """Gather a replay tree into `[batch, sequence, ...]` form."""
    if isinstance(value, dict):
        return {
            key: _tree_sequence(item, flat_idxs, batch_size, sequence_length)
            for key, item in value.items()
        }
    return value[flat_idxs].reshape(batch_size, sequence_length, *value.shape[1:])


def _tree_first_step(value):
    """Take the first sequence element from a `[batch, sequence, ...]` replay tree."""
    if isinstance(value, dict):
        return {key: _tree_first_step(item) for key, item in value.items()}
    return value[:, 0].copy()


def _tree_prefill(value, size: int, init_size: int):
    """Allocate a buffer tree and copy an initial dataset tree into it."""
    if isinstance(value, dict):
        return {key: _tree_prefill(item, size, init_size) for key, item in value.items()}
    value = np.asarray(value)
    buffer = np.zeros((size, *value.shape[1:]), dtype=value.dtype)
    buffer[:init_size] = value
    return buffer


def _tree_slice(value, stop: int):
    """Take the first `stop` entries from an array tree."""
    if isinstance(value, dict):
        return {key: _tree_slice(item, stop) for key, item in value.items()}
    return value[:stop]


def _assign_nested(root: dict[str, Any], key_path: str, value: np.ndarray) -> None:
    """Assign `value` into `root` using slash-separated npz keys."""
    parts = key_path.split("/")
    target = root
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def load_npz_dataset(path: str | Path, *, max_transitions: int | None = None) -> dict:
    """Load a replay dataset saved by `ReplayBuffer.save_npz`.

    Metadata keys such as `size`, `pointer`, `max_size`, and `frame_stack` are
    ignored. Nested arrays are reconstructed from slash-separated npz keys.
    """
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as npz:
        dataset = {}
        for key in npz.files:
            if key in {"size", "pointer", "max_size", "frame_stack"}:
                continue
            _assign_nested(dataset, key, npz[key])

    if max_transitions is not None:
        dataset = _tree_slice(dataset, int(max_transitions))
    if "observations" not in dataset or "actions" not in dataset:
        raise ValueError(f"NPZ replay dataset is missing required keys: {path}")

    # Keep old replay files compatible after residual-action metadata was added.
    actions = np.asarray(dataset["actions"])
    for key in ("base_actions", "residual_actions", "next_base_actions"):
        if key not in dataset:
            dataset[key] = np.full_like(actions, np.nan, dtype=np.float32)
    return dataset


def _flatten_tree(prefix: str, value, out: dict[str, np.ndarray]) -> None:
    """Flatten nested replay arrays for `np.savez_compressed`."""
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_tree(f"{prefix}/{key}", item, out)
        return
    out[prefix] = value


@dataclass
class EpisodeRecord:
    """Cached metadata for one complete episode stored in replay."""

    episode_id: int
    length: int
    indices: tuple[int, ...]


@dataclass
class ReplayBuffer:
    """Numpy replay buffer with QC-style n-step sequence sampling.

    `frame_stack` is intentionally supported as a constructor argument but kept
    at the default value of 1 for v0.  Image/frame-history handling can be
    expanded later without changing call sites.
    """

    data: dict[str, np.ndarray]
    max_size: int
    image_data: Any | None = None
    size: int = 0
    pointer: int = 0
    frame_stack: int = 1
    episode_records: dict[int, EpisodeRecord] = field(default_factory=dict)
    episode_worst_order: list[int] = field(default_factory=list)
    episode_index_dirty: bool = True

    @classmethod
    def create(cls, example_transition: dict, size: int, *, frame_stack: int = 1) -> "ReplayBuffer":
        """Allocate a zero-initialized circular buffer from one example transition."""
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1.")
        example_transition = dict(example_transition)
        observations, image_observations = _split_image_tree(example_transition["observations"])
        next_observations, _ = _split_image_tree(example_transition["next_observations"])
        example_transition["observations"] = observations
        example_transition["next_observations"] = next_observations
        data = {key: _tree_zeros(value, size) for key, value in example_transition.items()}
        image_data = None if _tree_empty(image_observations) else _tree_zeros(image_observations, size)
        return cls(data=data, max_size=size, image_data=image_data, frame_stack=frame_stack)

    @classmethod
    def create_from_initial_dataset(
        cls,
        init_dataset: dict,
        size: int,
        *,
        frame_stack: int = 1,
    ) -> "ReplayBuffer":
        """Create a replay buffer pre-filled with an offline/static dataset."""
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1.")
        init_size = get_size(init_dataset)
        if size < init_size:
            raise ValueError(f"size ({size}) must be >= initial dataset size ({init_size}).")
        init_dataset = dict(init_dataset)
        explicit_image_observations = init_dataset.pop("image_observations", None)
        observations, image_observations = _split_image_tree(init_dataset["observations"])
        if explicit_image_observations is not None:
            image_observations = explicit_image_observations
        next_observations, _ = _split_image_tree(init_dataset["next_observations"])
        init_dataset["observations"] = observations
        init_dataset["next_observations"] = next_observations
        data = {key: _tree_prefill(value, size, init_size) for key, value in init_dataset.items()}
        image_data = None
        if not _tree_empty(image_observations):
            image_data = _tree_prefill(image_observations, size, init_size)
        return cls(
            data=data,
            max_size=size,
            image_data=image_data,
            size=init_size,
            pointer=init_size % size,
            frame_stack=frame_stack,
        )

    def __getitem__(self, key: str) -> np.ndarray:
        """Return a raw replay array by key."""
        if key == "image_observations":
            return self.image_data
        return self.data[key]

    def items(self):
        """Iterate over raw replay array key/value pairs."""
        return self.data.items()

    def add_transition(self, transition: dict) -> None:
        """Insert one transition and enforce the fixed replay schema."""
        transition = dict(transition)
        observations, image_observations = _split_image_tree(transition["observations"])
        next_observations, _ = _split_image_tree(transition["next_observations"])
        transition["observations"] = observations
        transition["next_observations"] = next_observations

        missing = set(self.data) - set(transition)
        extra = set(transition) - set(self.data)
        if missing or extra:
            raise KeyError(f"Replay transition schema mismatch. missing={missing}, extra={extra}")
        if self.image_data is not None:
            _tree_assign(self.image_data, self.pointer, image_observations)
        elif not _tree_empty(image_observations):
            raise KeyError("ReplayBuffer was created without image observations.")

        for key, value in transition.items():
            _tree_assign(self.data[key], self.pointer, value)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        self.episode_index_dirty = True

    def add_episode(self, episode: list[dict]) -> int:
        """Append a full episode transition-by-transition."""
        if len(episode) > self.max_size:
            raise ValueError(f"Episode length {len(episode)} exceeds replay capacity {self.max_size}.")
        for transition in episode:
            self.add_transition(transition)
        return len(episode)

    def _rebuild_episode_index(self) -> None:
        """Rebuild the cached episode index sorted by worst-first length."""
        self.episode_records.clear()
        self.episode_worst_order.clear()
        if self.size == 0:
            self.episode_index_dirty = False
            return
        if "episode_ids" not in self.data:
            raise KeyError("ReplayBuffer has no `episode_ids`; episode-level replacement is unavailable.")
        episode_ids = np.asarray(self.data["episode_ids"][: self.size]).reshape(-1)
        for raw_episode_id in np.unique(episode_ids):
            episode_id = int(raw_episode_id)
            if episode_id < 0:
                continue
            indices = tuple(map(int, np.nonzero(episode_ids == episode_id)[0]))
            self.episode_records[episode_id] = EpisodeRecord(
                episode_id=episode_id,
                length=len(indices),
                indices=indices,
            )
        self.episode_worst_order = sorted(
            self.episode_records,
            key=lambda episode_id: (
                self.episode_records[episode_id].length,
                episode_id,
            ),
            reverse=True,
        )
        self.episode_index_dirty = False

    def _ensure_episode_index(self) -> None:
        """Materialize the episode index on first use or after replay mutation."""
        if self.episode_index_dirty:
            self._rebuild_episode_index()

    def episode_lengths(self) -> dict[int, int]:
        """Return stored episode lengths keyed by non-negative `episode_ids`."""
        self._ensure_episode_index()
        return {episode_id: record.length for episode_id, record in self.episode_records.items()}

    def episode_indices(self, episode_id: int) -> np.ndarray:
        """Return stored indices belonging to one episode id."""
        self._ensure_episode_index()
        record = self.episode_records.get(int(episode_id))
        if record is None:
            return np.asarray([], dtype=np.int64)
        return np.asarray(record.indices, dtype=np.int64)

    def longest_episode(self) -> tuple[int, int, np.ndarray] | None:
        """Return `(episode_id, length, indices)` for the longest stored episode."""
        self._ensure_episode_index()
        if not self.episode_worst_order:
            return None
        episode_id = self.episode_worst_order[0]
        record = self.episode_records[episode_id]
        return episode_id, record.length, np.asarray(record.indices, dtype=np.int64)

    def worst_episode(self) -> tuple[int, int, np.ndarray] | None:
        """Return the currently worst demo episode under the length metric."""
        return self.longest_episode()

    def _transition_at(self, index: int) -> dict:
        """Reconstruct one transition for buffer compaction."""
        transition = {key: _tree_item(value, index) for key, value in self.data.items()}
        if self.image_data is not None:
            transition["observations"] = _merge_tree(
                transition["observations"],
                _tree_item(self.image_data, index),
            )
        return transition

    def _clear(self) -> None:
        """Reset the populated part of the buffer while preserving allocated arrays."""
        _tree_zero_in_place(self.data)
        if self.image_data is not None:
            _tree_zero_in_place(self.image_data)
        self.size = 0
        self.pointer = 0
        self.episode_records.clear()
        self.episode_worst_order.clear()
        self.episode_index_dirty = False

    def replace_episode(self, episode_id: int, episode: list[dict]) -> int:
        """Remove one stored episode and append a replacement episode."""
        if len(episode) > self.max_size:
            raise ValueError(f"Episode length {len(episode)} exceeds replay capacity {self.max_size}.")
        remove_indices = set(map(int, self.episode_indices(episode_id)))
        if not remove_indices:
            return 0
        kept = [self._transition_at(index) for index in range(self.size) if index not in remove_indices]
        removed = len(remove_indices)
        self._clear()
        for transition in kept:
            self.add_transition(transition)
        self.add_episode(episode)
        self._rebuild_episode_index()
        return removed

    def insert_episode(
        self,
        episode: list[dict],
        *,
        mode: Literal["none", "append", "replace_longest_if_better"] = "append",
    ) -> dict[str, int | str | None]:
        """Insert an episode by appending or replacing the longest worse episode.

        `replace_longest_if_better` is intended for demo buffers whose entries
        are successful trajectories but not necessarily optimal. The current
        longest episode is replaced when the new episode is shorter. If there
        is no comparable stored episode yet, the new episode is appended.
        """
        if not episode:
            return {
                "status": "empty",
                "added": 0,
                "removed": 0,
                "skipped": 1,
                "old_length": None,
                "new_length": 0,
            }
        if mode == "none":
            return {
                "status": "skipped_none",
                "added": 0,
                "removed": 0,
                "skipped": 1,
                "old_length": None,
                "new_length": len(episode),
            }
        if mode == "append":
            added = self.add_episode(episode)
            return {
                "status": "appended",
                "added": added,
                "removed": 0,
                "skipped": 0,
                "old_length": None,
                "new_length": added,
            }
        if mode != "replace_longest_if_better":
            raise ValueError(f"Unknown episode insert mode: {mode!r}")

        new_length = len(episode)
        if new_length > self.max_size:
            raise ValueError(f"Episode length {new_length} exceeds replay capacity {self.max_size}.")
        longest = self.longest_episode()
        if longest is None:
            if self.size + new_length <= self.max_size:
                added = self.add_episode(episode)
                self._rebuild_episode_index()
                return {
                    "status": "appended_no_comparable_episode",
                    "added": added,
                    "removed": 0,
                    "skipped": 0,
                    "old_length": None,
                    "new_length": added,
                }
            return {
                "status": "skipped_no_episode_metadata",
                "added": 0,
                "removed": 0,
                "skipped": 1,
                "old_length": None,
                "new_length": new_length,
            }
        episode_id, old_length, _ = longest
        if new_length >= old_length:
            return {
                "status": "skipped_not_better",
                "added": 0,
                "removed": 0,
                "skipped": 1,
                "old_length": old_length,
                "new_length": new_length,
            }
        removed = self.replace_episode(episode_id, episode)
        return {
            "status": "replaced",
            "added": new_length,
            "removed": removed,
            "skipped": 0,
            "old_length": old_length,
            "new_length": new_length,
        }

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample independent single-step transitions uniformly."""
        idxs = np.random.randint(self.size, size=batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs: np.ndarray) -> dict[str, np.ndarray]:
        """Return replay entries at explicit indices."""
        batch = {key: _tree_index(value, idxs) for key, value in self.data.items()}
        if self.image_data is not None:
            image_observations = _tree_index(self.image_data, idxs)
            next_image_observations, next_valid = self._next_image_observations(idxs)
            batch["observations"] = _merge_tree(batch["observations"], image_observations)
            batch["next_observations"] = _merge_tree(batch["next_observations"], next_image_observations)
            batch["image_next_valid"] = next_valid.astype(np.float32)
        return batch

    def _next_image_observations(self, idxs: np.ndarray):
        """Return image frames for next observations via the `i + 1` transition slot."""
        if self.image_data is None:
            raise ValueError("ReplayBuffer has no image observations.")
        next_idxs = (idxs + 1) % self.max_size
        current_episode_ids = self.data["episode_ids"][idxs]
        next_episode_ids = self.data["episode_ids"][next_idxs]
        current_episode_steps = self.data["episode_steps"][idxs]
        next_episode_steps = self.data["episode_steps"][next_idxs]
        next_valid = (
            (next_idxs < self.size)
            & (current_episode_ids == next_episode_ids)
            & (next_episode_steps == current_episode_steps + 1)
        )
        next_images = _tree_index(self.image_data, next_idxs)
        return _tree_mask_invalid(next_images, next_valid), next_valid

    def sample_sequence(self, batch_size: int, sequence_length: int, discount: float) -> dict[str, np.ndarray]:
        """Sample QC-style n-step sequences.

        For `sequence_length=n`, `rewards[:, -1]` is the discounted n-step
        return from the start index, and `next_observations[:, -1]` is the
        bootstrap observation after the final sampled transition.

        `valid[:, -1]` is the critic-loss mask for the n-step target.  Matching
        qc_base, samples that cross an episode boundary inside the n-step window
        are kept in the returned batch but masked out instead of truncated into
        a shorter target.
        """
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        if self.size < sequence_length:
            raise ValueError(f"Buffer size {self.size} is smaller than sequence_length {sequence_length}.")
        if self.frame_stack != 1:
            raise NotImplementedError("frame_stack > 1 is reserved for a later implementation.")

        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        all_idxs = idxs[:, None] + np.arange(sequence_length)[None, :]
        flat_idxs = all_idxs.reshape(-1)

        observations = _tree_sequence(self.data["observations"], flat_idxs, batch_size, sequence_length)
        next_observations = _tree_sequence(
            self.data["next_observations"],
            flat_idxs,
            batch_size,
            sequence_length,
        )
        actions = self.data["actions"][flat_idxs].reshape(
            batch_size, sequence_length, *self.data["actions"].shape[1:]
        )
        raw_rewards = self.data["rewards"][flat_idxs].reshape(batch_size, sequence_length)
        raw_masks = self.data["masks"][flat_idxs].reshape(batch_size, sequence_length)
        raw_terminals = self.data["terminals"][flat_idxs].reshape(batch_size, sequence_length)
        image_next_valid = None
        if self.image_data is not None:
            image_observations = _tree_sequence(self.image_data, flat_idxs, batch_size, sequence_length)
            next_images, flat_next_valid = self._next_image_observations(flat_idxs)
            next_image_observations = _tree_sequence(
                next_images,
                np.arange(len(flat_idxs)),
                batch_size,
                sequence_length,
            )
            observations = _merge_tree(observations, image_observations)
            next_observations = _merge_tree(next_observations, next_image_observations)
            image_next_valid = flat_next_valid.reshape(batch_size, sequence_length).astype(np.float32)

        rewards = np.zeros((batch_size, sequence_length), dtype=np.float32)
        masks = np.ones((batch_size, sequence_length), dtype=np.float32)
        terminals = np.zeros((batch_size, sequence_length), dtype=np.float32)
        valid = np.ones((batch_size, sequence_length), dtype=np.float32)

        discount_powers = discount ** np.arange(sequence_length)
        rewards[:, 0] = raw_rewards[:, 0]
        masks[:, 0] = raw_masks[:, 0]
        terminals[:, 0] = raw_terminals[:, 0]
        for i in range(1, sequence_length):
            # Accumulate discounted rewards from the start state while carrying
            # the chain mask used by the eventual bootstrap term.
            rewards[:, i] = rewards[:, i - 1] + raw_rewards[:, i] * discount_powers[i]
            masks[:, i] = np.minimum(masks[:, i - 1], raw_masks[:, i])
            terminals[:, i] = np.maximum(terminals[:, i - 1], raw_terminals[:, i])
            valid[:, i] = 1.0 - terminals[:, i - 1]

        batch = {
            "observations": _tree_first_step(observations),
            "actions": actions,
            "masks": masks,
            "rewards": rewards,
            "terminals": terminals,
            "valid": valid,
            "next_observations": next_observations,
        }
        core_sequence_keys = {
            "observations",
            "actions",
            "masks",
            "rewards",
            "terminals",
            "next_observations",
        }
        for key, value in self.data.items():
            if key in core_sequence_keys:
                continue
            batch[key] = _tree_sequence(value, flat_idxs, batch_size, sequence_length)
        if image_next_valid is not None:
            batch["image_next_valid"] = image_next_valid
        return batch

    def save_npz(self, path: str | Path) -> None:
        """Persist only the populated part of the replay buffer."""
        path = Path(path)
        payload = {}
        for key, value in self.data.items():
            _flatten_tree(key, _tree_index(value, np.arange(self.size)), payload)
        if self.image_data is not None:
            _flatten_tree(
                "image_observations",
                _tree_index(self.image_data, np.arange(self.size)),
                payload,
            )
        payload["size"] = np.asarray(self.size, dtype=np.int64)
        payload["pointer"] = np.asarray(self.pointer, dtype=np.int64)
        payload["max_size"] = np.asarray(self.max_size, dtype=np.int64)
        payload["frame_stack"] = np.asarray(self.frame_stack, dtype=np.int64)
        np.savez_compressed(path, **payload)
