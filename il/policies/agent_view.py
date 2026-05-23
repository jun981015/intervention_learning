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
        """Sample one action proposal using the wrapped agent.

        Stochastic agents may expose `sample_actions_with_log_prob`.
        Deterministic agents such as TD3-BC only need `sample_actions`; in that
        case log-probability is not defined and is stored as NaN.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        obs = jnp.asarray(observation, dtype=jnp.float32)
        if hasattr(self.agent, "sample_actions_with_log_prob"):
            action, log_prob = self.agent.sample_actions_with_log_prob(obs, rng=rng)
            log_prob_np = np.asarray(log_prob, dtype=np.float32).reshape(-1)
            log_prob_value = float(log_prob_np[0]) if log_prob_np.size else float("nan")
        elif hasattr(self.agent, "sample_actions"):
            action = self.agent.sample_actions(obs, rng=rng)
            log_prob_value = float("nan")
        else:
            raise ValueError(
                f"Actor kind={self.kind!r} must expose sample_actions() or "
                "sample_actions_with_log_prob()."
            )

        action_np, info = self._format_action(np.asarray(action))
        return PolicyOutput(
            action=action_np,
            log_prob=log_prob_value,
            info={
                "kind": self.kind,
                "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "",
                **info,
            },
        )
