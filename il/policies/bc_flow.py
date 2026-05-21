from __future__ import annotations

"""Policy adapter for flow-matching BC agents in this repo's checkpoint layout."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from il.algo.bc.flow import BCFlowAgent
from il.utils.flax_utils import restore_agent_with_file
from il.utils.types import PolicyOutput


@dataclass
class BCFlowPolicy:
    """Wrap a `BCFlowAgent` with the shared policy interface."""

    agent: BCFlowAgent
    checkpoint_path: Path
    obs_dim: int
    action_dim: int
    full_action_dim: int
    horizon_length: int
    return_full_chunk: bool = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        config: dict[str, Any],
        obs_dim: int,
        action_dim: int,
        seed: int = 0,
        return_full_chunk: bool = False,
    ) -> "BCFlowPolicy":
        """Create this repo's BC flow agent and restore a matching checkpoint."""
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        config = dict(config)
        horizon_length = int(config["horizon_length"])
        full_action_dim = action_dim * horizon_length if config["action_chunking"] else action_dim

        ex_observations = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        ex_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)
        agent = BCFlowAgent.create(seed, ex_observations, ex_actions, config)
        agent = restore_agent_with_file(agent, checkpoint_path)

        return cls(
            agent=agent,
            checkpoint_path=checkpoint_path,
            obs_dim=int(obs_dim),
            action_dim=int(action_dim),
            full_action_dim=int(full_action_dim),
            horizon_length=horizon_length,
            return_full_chunk=return_full_chunk,
        )

    def _format_action(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return either the full chunk or the first primitive action."""
        action = np.asarray(action, dtype=np.float32)
        info = {}
        if self.return_full_chunk or self.full_action_dim == self.action_dim:
            return action.reshape(-1), info

        chunk = action.reshape(self.horizon_length, self.action_dim)
        info["full_action_chunk"] = chunk
        info["chunk_index"] = 0
        return chunk[0], info

    def sample_action(self, observation: np.ndarray, *, rng) -> PolicyOutput:
        """Sample one implicit-flow action proposal."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        observation = jnp.asarray(observation, dtype=jnp.float32)
        action, log_prob = self.agent.sample_actions_with_log_prob(observation, rng=rng)
        action_np, info = self._format_action(np.asarray(action))
        log_prob_np = np.asarray(log_prob, dtype=np.float32).reshape(-1)
        return PolicyOutput(
            action=action_np,
            log_prob=float(log_prob_np[0]) if log_prob_np.size else float("nan"),
            info={
                "checkpoint_path": str(self.checkpoint_path),
                **info,
            },
        )
