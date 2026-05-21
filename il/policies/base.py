from __future__ import annotations

from typing import Protocol

import numpy as np

from il.utils.types import PolicyOutput


class Policy(Protocol):
    """Minimal policy interface shared by learner and expert."""

    def sample_action(self, observation: np.ndarray, *, rng) -> PolicyOutput:
        """Return one action proposal for the given observation."""
        ...
