from __future__ import annotations

"""Named replay buffers and ratio-based mixed sampling."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np


def _tree_concat(values: list):
    """Concatenate matching array trees along the batch axis."""
    first = values[0]
    if isinstance(first, dict):
        keys = set(first)
        for value in values[1:]:
            if not isinstance(value, dict) or set(value) != keys:
                raise KeyError("Nested mixed batch schema mismatch.")
        return {key: _tree_concat([value[key] for value in values]) for key in keys}
    return np.concatenate(values, axis=0)


def _tree_batch_size(value) -> int:
    """Return leading batch size from an array tree."""
    if isinstance(value, dict):
        return _tree_batch_size(next(iter(value.values())))
    return value.shape[0]


def _tree_index(value, indices: np.ndarray):
    """Index an array tree by batch indices."""
    if isinstance(value, dict):
        return {key: _tree_index(item, indices) for key, item in value.items()}
    return value[indices]


def _concat_batches(batches: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate same-schema replay batches along the batch axis."""
    if not batches:
        raise ValueError("At least one batch is required.")
    keys = set(batches[0])
    for batch in batches[1:]:
        if set(batch) != keys:
            raise KeyError(f"Mixed batch schema mismatch. expected={keys}, got={set(batch)}")
    return {key: _tree_concat([batch[key] for batch in batches]) for key in keys}


def _allocate_counts(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Convert sampling fractions into integer counts that sum to `total`."""
    if total < 1:
        raise ValueError("total must be >= 1.")
    if not fractions:
        raise ValueError("At least one sampling fraction is required.")
    names = list(fractions)
    weights = np.asarray([float(fractions[name]) for name in names], dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError(f"Sampling fractions must be non-negative: {fractions}")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError(f"At least one sampling fraction must be positive: {fractions}")

    raw = weights / weight_sum * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def _as_buffer_mapping(buffers) -> dict[str, object]:
    """Convert supported buffer containers to a plain dictionary."""
    if hasattr(buffers, "as_dict"):
        return dict(buffers.as_dict())
    if isinstance(buffers, Mapping):
        return dict(buffers)
    raise TypeError("buffers must be a mapping or an object with as_dict().")


@dataclass
class ReplayBufferCollection:
    """Named replay buffers for online, intervention, and demo data streams."""

    online: object
    intervention: object
    demo: object

    def as_dict(self) -> dict[str, object]:
        """Return buffers keyed by their canonical names."""
        return {
            "online": self.online,
            "intervention": self.intervention,
            "demo": self.demo,
        }

    def get(self, name: str):
        """Return one named buffer."""
        buffers = self.as_dict()
        if name not in buffers:
            raise KeyError(f"Unknown replay buffer '{name}'. Available buffers: {sorted(buffers)}")
        return buffers[name]


@dataclass(frozen=True)
class MixedSamplingSpec:
    """Fractions describing how to mix named replay buffers."""

    fractions: Mapping[str, float]

    def counts(self, batch_size: int) -> dict[str, int]:
        """Return integer sample counts for each named buffer."""
        return _allocate_counts(batch_size, self.fractions)


@dataclass
class MixedReplaySampler:
    """Sample batches from multiple replay buffers according to a ratio spec."""

    buffers: Mapping[str, object]
    spec: MixedSamplingSpec
    shuffle: bool = True

    def __post_init__(self) -> None:
        """Normalize supported buffer containers to a dictionary."""
        self.buffers = _as_buffer_mapping(self.buffers)

    def _sample_parts(self, batch_size: int, sample_fn: Callable) -> list[dict[str, np.ndarray]]:
        """Sample non-empty batch parts from configured buffers."""
        parts = []
        for name, count in self.spec.counts(batch_size).items():
            if count == 0:
                continue
            if name not in self.buffers:
                raise KeyError(f"Sampling spec requested missing buffer '{name}'.")
            parts.append(sample_fn(self.buffers[name], count))
        return parts

    def _maybe_shuffle(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Shuffle mixed samples so source order does not leak into batches."""
        if not self.shuffle:
            return batch
        batch_size = _tree_batch_size(next(iter(batch.values())))
        permutation = np.random.permutation(batch_size)
        return {key: _tree_index(value, permutation) for key, value in batch.items()}

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample single-step transitions from the mixed buffers."""
        parts = self._sample_parts(batch_size, lambda buffer, count: buffer.sample(count))
        return self._maybe_shuffle(_concat_batches(parts))

    def sample_sequence(self, batch_size: int, sequence_length: int, discount: float) -> dict[str, np.ndarray]:
        """Sample n-step sequences from the mixed buffers."""
        parts = self._sample_parts(
            batch_size,
            lambda buffer, count: buffer.sample_sequence(count, sequence_length, discount),
        )
        return self._maybe_shuffle(_concat_batches(parts))
