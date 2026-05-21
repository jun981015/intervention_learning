from __future__ import annotations

"""Policy view for trainable agents.

This adapter lets a learner/expert agent participate in rollout through the
shared action-only policy interface while keeping update logic in `algo`.
"""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from il.utils.types import PolicyOutput


class AgentPolicyView:
    """Action-only wrapper around a live trainable agent."""

    def __init__(
        self,
        *,
        agent: Any,
        kind: str,
        checkpoint_path: Path | None,
        obs_dim: int,
        action_dim: int,
        horizon_length: int,
        action_chunking: bool,
    ):
        self.agent = agent
        self.kind = kind
        self.checkpoint_path = checkpoint_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon_length = horizon_length
        self.action_chunking = action_chunking

    def _format_action(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one primitive action and keep full chunks in `info`."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        full_action_dim = self.action_dim * self.horizon_length if self.action_chunking else self.action_dim
        if full_action_dim == self.action_dim:
            return action, {}
        chunk = action.reshape(self.horizon_length, self.action_dim)
        return chunk[0], {"full_action_chunk": chunk, "chunk_index": 0}

    def sample_action(self, observation: np.ndarray, *, rng) -> PolicyOutput:
        """Sample one action proposal using the wrapped agent."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        action, log_prob = self.agent.sample_actions_with_log_prob(
            jnp.asarray(observation, dtype=jnp.float32),
            rng=rng,
        )
        action_np, info = self._format_action(np.asarray(action))
        log_prob_np = np.asarray(log_prob, dtype=np.float32).reshape(-1)
        return PolicyOutput(
            action=action_np,
            log_prob=float(log_prob_np[0]) if log_prob_np.size else float("nan"),
            info={
                "kind": self.kind,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "",
                **info,
            },
        )
