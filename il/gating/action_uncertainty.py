from __future__ import annotations

"""Action-uncertainty intervention gate."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from il.gating.base import GateContext
from il.utils.types import ControllerId, GateDecision, GateReason, PolicyOutput


@dataclass
class ActionUncertaintyGate:
    """Start expert intervention when sampled action variance is high."""

    threshold: float
    source: str = "learner"
    estimator: str = "sample_variance"
    num_samples: int = 8
    score: str = "rms_std"
    intervention_prob: float = 1.0
    intervention_horizon: int = 1
    _remaining_steps: int = field(default=0, init=False)
    _last_info: dict[str, float | int | str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Validate gate hyperparameters."""
        if self.source not in {"learner", "expert", "base"}:
            raise ValueError("source must be one of {'learner', 'expert', 'base'}.")
        if self.estimator != "sample_variance":
            raise ValueError("ActionUncertaintyGate currently supports estimator='sample_variance'.")
        if self.score != "rms_std":
            raise ValueError("ActionUncertaintyGate currently supports score='rms_std'.")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2 for sample variance.")
        if not 0.0 <= self.intervention_prob <= 1.0:
            raise ValueError("intervention_prob must be in [0, 1].")
        if self.intervention_horizon < 1:
            raise ValueError("intervention_horizon must be >= 1.")

    def reset_episode(self) -> None:
        """Drop sticky intervention state at an episode boundary."""
        self._remaining_steps = 0
        self._last_info = {}

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
            reason=GateReason.ACTION_UNCERTAINTY,
            score=float(info.get("uncertainty_score", 0.0)),
            info=info,
        )

    def _sample_action_variance(self, context: GateContext) -> tuple[float, dict[str, float | int | str]]:
        """Estimate action-space variance by resampling one policy source."""
        actions = []
        for _ in range(self.num_samples):
            output = context.sample_policy(self.source)
            action = np.asarray(output.action, dtype=np.float32).reshape(-1)
            if action.size != context.action_dim:
                raise ValueError(
                    f"ActionUncertaintyGate expected primitive action_dim={context.action_dim}, "
                    f"got action shape {np.asarray(output.action).shape} from source={self.source!r}."
                )
            if not np.isfinite(action).all():
                raise ValueError(f"ActionUncertaintyGate sampled non-finite action from source={self.source!r}.")
            actions.append(action)

        samples = np.stack(actions, axis=0)
        variance = np.var(samples, axis=0)
        std = np.sqrt(np.maximum(variance, 0.0))
        uncertainty_score = float(np.sqrt(np.mean(variance)))
        return uncertainty_score, {
            "uncertainty_score": uncertainty_score,
            "action_variance_mean": float(np.mean(variance)),
            "action_variance_max": float(np.max(variance)),
            "action_std_mean": float(np.mean(std)),
            "action_std_max": float(np.max(std)),
            "num_samples": int(self.num_samples),
            "source": self.source,
            "estimator": self.estimator,
            "score_kind": self.score,
        }

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
        context: GateContext | None = None,
    ) -> GateDecision:
        """Route to expert probabilistically when sampled action variance is high."""
        del step, observation, learner, expert, expert_agent, action_dim
        if self._remaining_steps > 0:
            return self._horizon_decision()
        if context is None:
            raise ValueError("ActionUncertaintyGate requires GateContext for policy resampling.")

        uncertainty_score, uncertainty_info = self._sample_action_variance(context)
        signal = uncertainty_score > self.threshold
        sample = float(rng.random())
        start_intervention = bool(signal and sample < self.intervention_prob)
        if start_intervention:
            self._remaining_steps = self.intervention_horizon - 1

        info = {
            **uncertainty_info,
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
        self._last_info = {
            key: value for key, value in info.items() if isinstance(value, (int, float, str))
        }
        return GateDecision(
            controller_id=ControllerId.EXPERT if start_intervention else ControllerId.LEARNER,
            reason=GateReason.ACTION_UNCERTAINTY,
            score=uncertainty_score,
            info=info,
        )
