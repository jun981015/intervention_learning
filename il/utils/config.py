from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level online training knobs not owned by a specific algorithm."""

    batch_size: int = 256
    utd_ratio: int = 1
    horizon_length: int = 1
    discount: float = 0.99
    sampling_fractions: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate training knobs that must be positive."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.utd_ratio < 1:
            raise ValueError("utd_ratio must be >= 1.")
        if self.horizon_length < 1:
            raise ValueError("horizon_length must be >= 1.")
        if self.sampling_fractions is not None and not self.sampling_fractions:
            raise ValueError("sampling_fractions must not be empty when provided.")


@dataclass(frozen=True)
class DAggerConfig:
    """DAgger-specific rollout options."""

    store_expert_action: bool = True
