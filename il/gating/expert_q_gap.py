from __future__ import annotations

"""Expert-Q gap intervention gate."""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput


@dataclass
class ExpertQGapGate:
    """Start an expert intervention when expert-Q prefers expert action.

    The gate compares two actions at the same state under the loaded expert
    critic:

        gap = Q_expert(s, a_expert) - Q_expert(s, a_learner)

    If `gap > threshold`, expert control starts with probability
    `intervention_prob`. Once started, the expert controls the environment for
    `intervention_horizon` consecutive env steps including the trigger step.
    """

    threshold: float
    intervention_prob: float
    intervention_horizon: int = 1
    q_agg: str = "min"
    _remaining_steps: int = field(default=0, init=False)
    _last_info: dict[str, float | int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Validate gate hyperparameters."""
        if not 0.0 <= self.intervention_prob <= 1.0:
            raise ValueError("intervention_prob must be in [0, 1].")
        if self.intervention_horizon < 1:
            raise ValueError("intervention_horizon must be >= 1.")
        if self.q_agg not in ("min", "mean", "max"):
            raise ValueError("q_agg must be one of {'min', 'mean', 'max'}.")

    def reset_episode(self) -> None:
        """Drop sticky intervention state at an episode boundary."""
        self._remaining_steps = 0
        self._last_info = {}

    def _agent_config_get(self, expert_agent: Any, key: str, default: Any) -> Any:
        """Read an optional config key without assuming a concrete agent class."""
        config = getattr(expert_agent, "config", None)
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    def _action_for_critic(self, output: PolicyOutput, *, expert_agent: Any, action_dim: int) -> np.ndarray:
        """Format a policy proposal into the action shape expected by expert Q."""
        action = np.asarray(output.action, dtype=np.float32).reshape(-1)
        action_chunking = bool(self._agent_config_get(expert_agent, "action_chunking", False))
        horizon_length = int(self._agent_config_get(expert_agent, "horizon_length", 1))
        full_dim = action_dim * horizon_length if action_chunking else action_dim

        if action.size == full_dim:
            return action.reshape(1, full_dim)
        if action_chunking and "full_action_chunk" in output.info:
            chunk = np.asarray(output.info["full_action_chunk"], dtype=np.float32)
            return chunk.reshape(1, full_dim)
        if not action_chunking and action.size == action_dim:
            return action.reshape(1, action_dim)
        raise ValueError(
            "ExpertQGapGate cannot format action for the expert Q function: "
            f"action_size={action.size}, expected={full_dim}, action_chunking={action_chunking}. "
            "If the expert Q uses action chunks, policy output must include `full_action_chunk`."
        )

    def _expert_q_value(self, expert_agent: Any, observations: jnp.ndarray, actions: jnp.ndarray):
        """Call the expert's explicit scalar action-value API."""
        if hasattr(expert_agent, "evaluate_q"):
            return expert_agent.evaluate_q(observations, actions, q_agg=self.q_agg)
        if hasattr(expert_agent, "q_values"):
            return expert_agent.q_values(observations, actions, q_agg=self.q_agg)
        raise ValueError(
            "ExpertQGapGate requires an action-value expert API. "
            "The expert agent must expose `evaluate_q(obs, action, q_agg=...)` or "
            "`q_values(obs, action, q_agg=...)`. Value-only PPO critics cannot compute "
            "Q(s, a_expert) - Q(s, a_learner) without an action-value head."
        )

    def _critic_value(self, expert_agent: Any, observation: np.ndarray, action: np.ndarray) -> float:
        """Evaluate one action with the expert's scalar Q API."""
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        q_value = self._expert_q_value(expert_agent, jnp.asarray(obs), jnp.asarray(action, dtype=jnp.float32))
        q_np = np.asarray(q_value, dtype=np.float32).reshape(-1)
        if q_np.size != 1:
            raise ValueError(
                "Expert scalar Q API must return exactly one value for one observation/action pair. "
                f"Got shape={np.asarray(q_value).shape}."
            )
        if not bool(np.isfinite(q_np).all()):
            raise ValueError(
                "Expert scalar Q API returned a non-finite value. This usually means an action "
                "proposal was missing or invalid, e.g. expert_query was not set to 'always' "
                "for expert_q_gap."
            )
        return float(q_np[0])

    def _horizon_decision(self) -> GateDecision:
        """Continue an already-started intervention segment."""
        self._remaining_steps -= 1
        info = {
            **self._last_info,
            "signal": 1,
            "intervention_started": 0,
            "intervention": 1,
            "intervention_prob": self.intervention_prob,
            "intervention_horizon": self.intervention_horizon,
            "horizon_active": 1,
            "remaining_steps": self._remaining_steps,
        }
        return GateDecision(
            controller_id=ControllerId.EXPERT,
            reason=GateReason.EXPERT_Q_GAP,
            score=float(info.get("q_gap", 0.0)),
            info=info,
        )

    def decide(
        self,
        *,
        step: int,
        observation: np.ndarray,
        learner: PolicyOutput,
        expert: PolicyOutput,
        rng: np.random.Generator,
        expert_agent: Any | None = None,
        action_dim: int | None = None,
    ) -> GateDecision:
        """Route to expert probabilistically when expert-Q gap is large."""
        del step
        if self._remaining_steps > 0:
            return self._horizon_decision()
        if expert_agent is None:
            raise ValueError("ExpertQGapGate requires the loaded expert agent.")
        if action_dim is None:
            raise ValueError("ExpertQGapGate requires action_dim.")

        expert_action = self._action_for_critic(expert, expert_agent=expert_agent, action_dim=action_dim)
        learner_action = self._action_for_critic(learner, expert_agent=expert_agent, action_dim=action_dim)
        q_expert = self._critic_value(expert_agent, observation, expert_action)
        q_learner = self._critic_value(expert_agent, observation, learner_action)
        q_gap = q_expert - q_learner
        signal = q_gap > self.threshold
        sample = float(rng.random())
        start_intervention = bool(signal and sample < self.intervention_prob)
        if start_intervention:
            self._remaining_steps = self.intervention_horizon - 1

        info = {
            "q_expert": q_expert,
            "q_learner": q_learner,
            "q_gap": q_gap,
            "threshold": self.threshold,
            "signal": int(signal),
            "random_sample": sample,
            "intervention_prob": self.intervention_prob,
            "intervention_horizon": self.intervention_horizon,
            "intervention_started": int(start_intervention),
            "intervention": int(start_intervention),
            "horizon_active": 0,
            "remaining_steps": self._remaining_steps,
        }
        self._last_info = {key: value for key, value in info.items() if isinstance(value, (int, float))}
        return GateDecision(
            controller_id=ControllerId.EXPERT if start_intervention else ControllerId.LEARNER,
            reason=GateReason.EXPERT_Q_GAP,
            score=q_gap,
            info=info,
        )
